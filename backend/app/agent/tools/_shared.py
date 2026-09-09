"""Infra shared by every tool submodule (call recording, the per-turn
budget, untrusted-data wrapping, API-message surfacing, scope labeling)
plus the tools that don't funnel through _build_filters: search_guide,
lookup_agency, get_agency_budget, get_agency_award_breakdown,
list_top_agencies_by_budget.
"""
from __future__ import annotations

import contextvars
import logging

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    AgencySubAgencyResponse,
    AgencyYearBudget,
    ObligationByPeriod,
    ToptierAgency,
    USASpendingAPIError,
)

from ..singletons import (
    RERANK_CONFIDENCE_THRESHOLD,
    _get_retriever,
    _get_usaspending_client,
)
from ..tool_filters import (
    AWARD_TYPE_GROUPS,
    AwardType,
    _clamp_limit,
    _normalize_award_type,
)

logger = logging.getLogger(__name__)

# Per-request capture buffer for structured tool results, so chart-worthy
# data survives past the @beta_tool wrapper that only returns a string to
# the LLM. A plain module-level list would leak between concurrent requests
# — main.py's /ask is a sync endpoint, which FastAPI runs on a real OS
# thread pool, so concurrent requests genuinely run at the same time.
# contextvars.ContextVar gives each call to ask() its own isolated buffer
# automatically, the same mechanism LangSmith's own tracing relies on.
_tool_call_log: contextvars.ContextVar[list[tuple[str, object, dict]] | None] = contextvars.ContextVar(
    "tool_call_log", default=None
)


def _record_tool_call(tool_name: str, result: object, context: dict | None = None) -> None:
    """Append to the current call's capture buffer, if one is active (set by
    ask() before starting the tool loop). No-op outside ask() - e.g. a tool
    function invoked directly, as the dev_tools scripts and tests do.

    context carries call-specific info the structured result itself doesn't
    include (e.g. agency_name, since the API response doesn't echo back the
    filters it was queried with) - used for things like chart titles that
    need to distinguish multiple calls to the same tool in one turn.
    """
    log = _tool_call_log.get()
    if log is not None:
        log.append((tool_name, result, context or {}))


# 15 comfortably covers a legitimate multi-agency comparison in one
# question while still bounding runaway/abusive tool use. Only counts the
# data tools (everything that calls _record_tool_call) - arithmetic tools
# never touch _tool_call_log, so a math-heavy question doesn't burn this
# budget on free, local computation.
MAX_TOOL_CALLS_PER_TURN = 15


def _check_tool_call_budget() -> str | None:
    """Checked at the very top of every data tool, before any real work
    (a live API call) happens - gating in _record_tool_call itself would
    be too late, since by the time a tool calls that the expensive part
    is already done. Returns an error string to hand back to the model
    (so it degrades to "answer with what you have," not a crash) if this
    turn already hit the cap, else None to proceed normally.
    """
    log = _tool_call_log.get()
    if log is not None and len(log) >= MAX_TOOL_CALLS_PER_TURN:
        return (
            f"Tool call budget exceeded for this turn ({MAX_TOOL_CALLS_PER_TURN} calls "
            "already made). Answer using what you've already retrieved, or tell the user "
            "to ask a narrower question covering fewer agencies/breakdowns at once."
        )
    return None


def _wrap_untrusted(text: str) -> str:
    """Mark tool-returned content as data, not instructions.

    Retrieved guide/glossary text and live API fields (agency names,
    category labels, mission statements, recipient names) are real-world
    text this app doesn't control or validate. This is defense-in-depth,
    not a fix for a demonstrated gap - the model already resists a naive
    "ignore instructions" payload without any wrapping - but making the
    instruction/data boundary explicit costs nothing. See
    _build_system_prompt for the matching instruction.
    """
    return f"<untrusted_data>\n{text}\n</untrusted_data>"


def _truncation_note(has_next: bool, shown: int) -> str:
    """A plain-language caveat appended to get_spending_by_category's and
    search_awards's formatted output when the live API's
    page_metadata.hasNext says more results exist beyond what `limit`
    returned - without it, the model can present a partial slice as the
    complete list.

    Appended directly into the tool's own returned text, not just
    mentioned in the system prompt, since the model follows an inline
    instruction not to overstate a result it just received far more
    reliably than a general system-prompt rule.
    """
    if not has_next:
        return ""
    return (
        f"\n\n(Note: this shows the top {shown} by amount - more results match these "
        "filters but aren't shown here. Do not present this as the complete or "
        "exhaustive list; tell the user more results exist. A larger limit or a "
        "narrower filter, e.g. a tighter min_amount, would show more of them.)"
    )


def _format_api_messages(messages: list[str] | None) -> str:
    """Surfaces the live API's own `messages` field - it explicitly flags
    any filter combination the request couldn't use (e.g. "The following
    filters from the request were not used: {'recipient_id'}...") instead
    of that happening without a trace."""
    if not messages:
        return ""
    return "\n\n(API notice: " + " ".join(messages) + ")"


def _scope_label(
    agency_name: str | None,
    recipient_name: str | None,
    recipient_id: str | None,
    *,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
    keywords: str | None = None,
) -> str:
    """Label for a failure/no-results message. _build_filters guarantees
    at least one of these is set."""
    return (
        agency_name or recipient_name or recipient_id
        or performed_in_state or recipient_in_state
        or naics_code or psc_code or cfda_program or keywords
        or "unknown scope"
    )


def _record_code_execution_calls(message) -> None:
    """Record any bash_code_execution calls in this message into the same
    capture buffer as every other tool, for citation purposes.

    code_execution is Anthropic's server-side tool, not one of our own
    @beta_tool functions - there's no function of ours in the call path to
    put a _record_tool_call() line inside, unlike every other tool here.
    Both the tool_use and its result appear in the same message's content
    list for a server tool (no client round-trip the way our own tools
    work), so this can be checked per-message rather than needing to track
    pending calls across turns.

    Only bash_code_execution is handled - text_editor_code_execution (file
    view/create/edit) calls aren't recorded, since this fallback's intended
    use (multi-step math, statistics none of the six typed tools compute)
    is expected to run as Bash/Python commands, not file edits.
    """
    tool_use_blocks = {b.id: b for b in message.content if getattr(b, "type", None) == "server_tool_use"}
    for block in message.content:
        if getattr(block, "type", None) != "bash_code_execution_tool_result":
            continue
        tool_use = tool_use_blocks.get(block.tool_use_id)
        if tool_use is None or tool_use.name != "bash_code_execution":
            continue
        command = tool_use.input.get("command", "")
        _record_tool_call("code_execution", block.content, {"command": command})


@beta_tool
def search_guide(query: str) -> str:
    """Search the Analyst's Guide to Federal Spending Data and the USASpending Glossary for conceptual or definitional information about USASpending — what a term means, how a data element is defined, which fields contain what.

    Args:
        query: What to search for in the guide or glossary.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    results = _get_retriever().retrieve(query, top_k=3)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]
    if not matches:
        return "No relevant content found in the Analyst's Guide or Glossary for this query."

    _record_tool_call("search_guide", matches)

    def _label(m: dict) -> str:
        # Glossary chunks carry a term and no real page number; Guide
        # chunks carry a page and no term - see ingest_glossary.py.
        return f"[{m['source']}: {m['term']}]" if m.get("term") else f"[Page {m['page_start']}]"

    return _wrap_untrusted("\n\n---\n\n".join(f"{_label(m)}\n{m['text']}" for m in matches))


@beta_tool
def lookup_agency(name: str) -> str:
    """Look up a federal agency by name to get its basic profile: toptier code, abbreviation, mission, website, Congressional Justification of Budget link, and subtier agency count. Use this for questions about what a specific agency is or does, or as a first step before any spending-data question that needs an agency's toptier code.

    Args:
        name: The agency name to search for, e.g. "National Science Foundation" or "NSF".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(name)
    if agency is None:
        logger.warning("lookup_agency: no agency found matching %r", name)
        return f"No agency found matching '{name}'."

    overview = client.get_agency_overview(agency.toptier_code)
    _record_tool_call("lookup_agency", overview, {"name": name})
    return _wrap_untrusted(
        f"Agency: {overview.name} ({overview.abbreviation})\n"
        f"Toptier code: {overview.toptier_code}\n"
        f"Fiscal year: {overview.fiscal_year}\n"
        f"Subtier agency count: {overview.subtier_agency_count}\n"
        f"Mission: {overview.mission or 'N/A'}\n"
        f"Website: {overview.website or 'N/A'}\n"
        f"Congressional Justification of Budget: {overview.congressional_justification_url or 'N/A'}"
    )


@traceable(run_type="tool", name="get_agency_budget_raw")
def get_agency_budget_raw(
    agency_name: str, start_fiscal_year: int, end_fiscal_year: int
) -> list[AgencyYearBudget]:
    """Call the API once (it always returns every fiscal year it has - no
    range param exists), return only the years actually asked for. Raises
    USASpendingAPIError if agency_name doesn't resolve, same pattern as
    the other tools' _raw functions.
    """
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")

    response = client.get_agency_budgetary_resources(agency.toptier_code)
    return [
        y for y in response.agency_data_by_year
        if start_fiscal_year <= y.fiscal_year <= end_fiscal_year
    ]


# P01 = October, the first month of the federal fiscal year - period N
# maps to this list's index N-1.
_FISCAL_PERIOD_MONTHS = [
    "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
]


def _format_period_breakdown(periods: list[ObligationByPeriod]) -> str:
    """Each period's `obligated` is cumulative from the start of the fiscal
    year through that period, not an incremental per-period amount - the
    label says so explicitly rather than trusting the model to infer it,
    the same reasoning as search_awards's cumulative Award Amount caveat.
    """
    ordered = sorted(periods, key=lambda p: p.period)
    parts = [f"{_FISCAL_PERIOD_MONTHS[p.period - 1]} ${p.obligated:,.2f}" for p in ordered]
    return "Obligated, cumulative from the start of the fiscal year: " + ", ".join(parts)


@beta_tool
def get_agency_budget(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    include_period_breakdown: bool = False,
) -> str:
    """Get an agency's actual appropriated budgetary resources, obligations, and outlays for a fiscal year range. Use this specifically for "what is X's budget," "how much money does X have," or "how much has X actually paid out" questions.

    This is a genuinely different concept from what get_spending_by_category, get_spending_over_time, and search_awards report: those three track money obligated against specific contracts, grants, and loans (award-level spending activity), not the agency's appropriated budget authority. An agency's total budgetary resources for a fiscal year is NOT the same number as its total award spending in that year, and the two should never be presented as if interchangeable - if asked about budget/appropriations specifically, use this tool, not the spending tools, even though both involve dollar figures for the same agency.

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021. This endpoint's
            own data only goes back to FY2017 (a shorter history than the FY2008 floor the
            spending tools have) - a range starting earlier than that will just return whatever
            years are actually available, not error.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        include_period_breakdown: Set True only when the question is specifically about WHEN
            during the fiscal year money was obligated (e.g. "how did NSF's obligations build up
            over FY2024" or "was most of the budget obligated early or late in the year") - false
            by default since most budget questions just want the yearly totals, and the
            breakdown adds up to ~11 extra numbers per fiscal year. Each period's obligated
            amount is CUMULATIVE from the start of that fiscal year through that period, not an
            incremental amount for that period alone - e.g. period 6 is the running total through
            that point in the year, not what was newly obligated in period 6. If you state what
            share of the year's total a period represents, or how much changed between two
            periods, call percentage_of or delta for that number - do not compute it yourself
            just because it looks like simple subtraction or a percentage of a total you can
            already see in this data.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        years = get_agency_budget_raw(agency_name, start_fiscal_year, end_fiscal_year)
    except USASpendingAPIError as e:
        logger.warning("get_agency_budget failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."

    _record_tool_call(
        "get_agency_budget",
        years,
        {"agency_name": agency_name, "start_fiscal_year": start_fiscal_year, "end_fiscal_year": end_fiscal_year},
    )

    if not years:
        return f"No budget data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year} (this endpoint's data starts at FY2017)."

    def _fmt(amount: float | None) -> str:
        return f"${amount:,.2f}" if amount is not None else "not reported"

    lines = []
    for y in sorted(years, key=lambda y: y.fiscal_year):
        line = (
            f"FY{y.fiscal_year}: budgetary resources {_fmt(y.agency_budgetary_resources)}, "
            f"obligated {_fmt(y.agency_total_obligated)}, outlayed {_fmt(y.agency_total_outlayed)}"
        )
        if include_period_breakdown and y.agency_obligation_by_period:
            line += "\n  " + _format_period_breakdown(y.agency_obligation_by_period)
        lines.append(line)
    # total_budgetary_resources (government-wide, not this agency's figure -
    # see AgencyYearBudget's docstring) is deliberately never included here.
    return _wrap_untrusted("\n".join(lines))


@beta_tool
def _format_top_agencies_by_budget(ranked: list[ToptierAgency]) -> str:
    period_label = f"FY{ranked[0].active_fy} Q{ranked[0].active_fq}" if ranked else "current period"
    lines = [
        f"{i}. {a.agency_name} ({a.abbreviation}): ${a.budget_authority_amount:,.2f} "
        f"({a.percentage_of_total_budget_authority:.2%} of total federal budget authority)"
        for i, a in enumerate(ranked, start=1)
    ]
    return f"As of {period_label}:\n" + "\n".join(lines)


@beta_tool
def list_top_agencies_by_budget(limit: int = 10) -> str:
    """List federal agencies ranked by budgetary resources (budget authority), largest first, each with its share of the total federal budget. Use this for "which agency has the biggest budget" or "what percent of the federal budget does X account for" questions.

    Always reflects the current fiscal year/quarter (the live data this is sourced from has no historical fiscal-year parameter) - the output states which period the figures are for. Use get_agency_budget instead for a specific agency's budget history across past fiscal years.

    Args:
        limit: Max number of agencies to return (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    limit = _clamp_limit(limit)
    client = _get_usaspending_client()
    agencies = client.list_toptier_agencies()
    ranked = sorted(agencies, key=lambda a: a.budget_authority_amount, reverse=True)[:limit]

    _record_tool_call("list_top_agencies_by_budget", ranked, {"limit": limit})

    return _wrap_untrusted(_format_top_agencies_by_budget(ranked))


@traceable(run_type="tool", name="get_agency_award_breakdown_raw")
def get_agency_award_breakdown_raw(
    agency_name: str,
    fiscal_year: int,
    award_type: AwardType | None = None,
) -> AgencySubAgencyResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError if agency_name doesn't resolve."""
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")
    award_type_codes = AWARD_TYPE_GROUPS[_normalize_award_type(award_type)] if award_type else None
    return client.get_agency_sub_agency_breakdown(
        agency.toptier_code, fiscal_year=fiscal_year, award_type_codes=award_type_codes, limit=50,
    )


def _format_agency_award_breakdown(response: AgencySubAgencyResponse) -> str:
    return "\n".join(
        f"{r.name}{f' ({r.abbreviation})' if r.abbreviation else ''}: "
        f"${r.total_obligations:,.2f} across {r.transaction_count:,} transactions, "
        f"{r.new_award_count:,} new awards"
        for r in response.results
    )


@beta_tool
def get_agency_award_breakdown(
    agency_name: str,
    fiscal_year: int,
    award_type: AwardType | None = None,
) -> str:
    """Get one agency's award spending broken down by sub-agency for a single fiscal year, including transaction counts and new-award counts alongside the dollar totals — not just the amount get_spending_by_category(category="awarding_subagency") gives. Use this specifically when the question asks about counts (how many transactions, how many new awards), not just dollar amounts.

    This is a genuinely different endpoint from get_spending_by_category and get_agency_budget, not a formatting variant of either: use get_spending_by_category instead for a dollar-only breakdown or a breakdown by anything other than sub-agency (NAICS, PSC, recipient, etc. — it has no count fields at all), and get_agency_budget instead for the agency's appropriated budget authority (a different number from award obligations). This tool is always scoped to exactly one agency and one fiscal year — never a recipient, never a range.

    Args:
        agency_name: The agency's name, e.g. "Department of Health and Human Services".
        fiscal_year: A single fiscal year, e.g. 2024 for FY2024 — this endpoint does not accept a
            range; call again for each year if a multi-year breakdown is needed.
        award_type: Optional. Restrict to one award type or bucket — same vocabulary as
            search_awards's award_type. Omit to include all award types.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_agency_award_breakdown_raw(agency_name, fiscal_year, award_type)
    except USASpendingAPIError as e:
        logger.warning("get_agency_award_breakdown failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."

    _record_tool_call(
        "get_agency_award_breakdown",
        response,
        {"agency_name": agency_name, "fiscal_year": fiscal_year, "award_type": award_type},
    )

    if not response.results:
        return f"No award data found for {agency_name} in FY{fiscal_year}."

    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
    return _wrap_untrusted(_format_agency_award_breakdown(response) + note)

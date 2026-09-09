"""The @beta_tool-decorated functions the agent calls, and their _raw
variants (structured Pydantic responses, presentation-free) that also feed
should_chart/build_tool_citation after the tool-calling loop finishes.

The _raw variants (not the @beta_tool wrappers) are @traceable. Tried
stacking @traceable directly under @beta_tool first and it broke tool
schemas: traceable injects its own optional `config` kwarg into the
wrapped function's signature, and beta_tool's schema generation picked it
up as a real, model-visible tool parameter. The _raw functions were never
@beta_tool in the first place, so tracing them is risk-free and also puts
the trace where the actual work (agency resolution, API calls) happens,
not on the thin formatting wrapper around it.
"""
from __future__ import annotations

import contextvars
import logging
from typing import Any, Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    AgencyYearBudget,
    IDVAmountsResponse,
    ObligationByPeriod,
    RecipientListing,
    RecipientLocation,
    RecipientOverview,
    SearchAwardsResponse,
    SpendingByCategoryResponse,
    SpendingOverTimeResponse,
    USASpendingAPIError,
)

from .response_shaping import _format_time_period
from .singletons import (
    RERANK_CONFIDENCE_THRESHOLD,
    _get_retriever,
    _get_usaspending_client,
)
from .tool_filters import (
    SEARCH_AWARDS_FIELDS_BASE,
    AwardType,
    DateType,
    RecipientAwardType,
    Scope,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
    _normalize_recipient_award_type,
    _record_optional_filter_context,
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


# Found live via red-teaming (BACKLOG.md "Red team: resource abuse"):
# "look up the toptier code for NSF, NASA, EPA, DOE, and DOD" triggered 5
# real, uncapped API calls in one turn - nothing stopped a much longer
# list. 15 is generous for any legitimate multi-agency comparison a real
# analyst would ask in one question, while still bounding a pathological
# one. Only counts the six data tools (everything that calls
# _record_tool_call) - the arithmetic tools never touch _tool_call_log at
# all, so a math-heavy question doesn't burn this budget on free, local
# computation that was never the actual resource-abuse surface.
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
    not a fix for a demonstrated gap - red-teaming already found the model
    resists a naive "ignore instructions" payload embedded in this kind of
    content without any wrapping - but making the instruction/data
    boundary explicit costs nothing. See _build_system_prompt for the
    matching instruction.
    """
    return f"<untrusted_data>\n{text}\n</untrusted_data>"


def _truncation_note(has_next: bool, shown: int) -> str:
    """A plain-language caveat appended to get_spending_by_category's and
    search_awards's formatted output when the live API's
    page_metadata.hasNext says more results exist beyond what `limit`
    returned.

    Found live (2026-09-06): a min_amount-filtered search_awards query
    silently returned 5 of a larger real match set (page_metadata.hasNext
    was true), and the model presented that partial slice as the complete
    list of matching awards - the exact "confident but incomplete" failure
    the filter layer above this was built to close, one layer further in.
    hasNext was already in the live API response the whole time; it just
    wasn't being read - client.search_awards/spending_by_category
    previously discarded page_metadata entirely.

    Appending this directly into the tool's own returned text (not just a
    docstring warning) matches this file's existing pattern for steering
    the model inline - see get_spending_by_category's failure message,
    which tells the model what not to do in the returned string itself,
    not just in the system prompt.
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
    """Surfaces the live API's own `messages` field - found live 2026-09-08
    via a raw curl to spending_by_award while investigating whether
    recipient_id silently no-ops there: it doesn't fail silently at all,
    it says exactly what happened ("The following filters from the
    request were not used: {'recipient_id'}..."). SpendingByCategoryResponse
    and SpendingOverTimeResponse already modeled this field; nothing ever
    read it. Generalizes past that one case - this fires for any filter
    combination the live API itself flags as unused/invalid, on any of
    the three tools, including ones not specifically anticipated here."""
    if not messages:
        return ""
    return "\n\n(API notice: " + " ".join(messages) + ")"


def _scope_label(agency_name: str | None, recipient_name: str | None, recipient_id: str | None) -> str:
    """What to call the query's scope in a human-facing message (a failure
    string, a "no results" message) when agency_name may now be absent -
    _build_filters (tool_filters.py) guarantees at least one of these
    three is set, so this always has something real to show."""
    return agency_name or recipient_name or recipient_id or "unknown scope"


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
    is expected to run as Bash/Python commands, not file edits. If that
    assumption turns out wrong, file operations would silently go uncited -
    worth revisiting if it comes up.
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
    """Look up a federal agency by name to get its basic profile: toptier code, abbreviation, mission, website, and subtier agency count. Use this for questions about what a specific agency is or does, or as a first step before any spending-data question that needs an agency's toptier code.

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
        f"Website: {overview.website or 'N/A'}"
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


# P01 = October, the first month of the federal fiscal year (verified
# against the local Glossary's "Submission Period" entry) - period N maps to
# this list's index N-1.
_FISCAL_PERIOD_MONTHS = [
    "Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep",
]


def _format_period_breakdown(periods: list[ObligationByPeriod]) -> str:
    """Each period's `obligated` is cumulative from the start of the fiscal
    year through that period (verified live 2026-09-07: period 12's value
    exactly equals the year's agency_total_obligated), not an incremental
    per-period amount - the label says so explicitly rather than trusting
    the model to infer it, the same reasoning as search_awards's cumulative
    Award Amount caveat.
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


# Verified live against the real API (checked 2026-09-06/07), not the API
# contract's own documented list - the contract lists 18 categories, but 4
# of them (object_class, program_activity, recipient_parent_duns, tas) 404
# in practice despite being documented (see BACKLOG.md's "Daily health
# check" entry). Same "code owns the exact vocabulary, not the model"
# pattern as AWARD_TYPE_GROUPS/US_STATE_ABBREVIATIONS - this used to be a
# bare str parameter with the valid values only in docstring prose, the
# same anti-pattern the award_type fix (see AWARD_TYPE_GROUPS's own
# comment) closed elsewhere. Wasn't dangerous yet (an unrecognized value
# 404s cleanly rather than silently substituting), but it's the same
# fragility, just not yet exploited.
#
# "recipient" is real and live-verified (re-confirmed 2026-09-07) but
# isn't in the API contract's own top-level category enum - found via the
# live category subdirectory listing during the original audit. Kept
# alongside recipient_duns (which returns identical results for every
# case tested) rather than instead of it, since recipient additionally
# carries a uei field and models a "MULTIPLE RECIPIENTS" rollup row that
# recipient_duns's schema doesn't.
VALID_CATEGORIES = {
    "awarding_agency", "awarding_subagency", "cfda", "country", "county",
    "defc", "district", "federal_account", "funding_agency",
    "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
    "state_territory",
}

# Static Literal mirror of VALID_CATEGORIES, used as get_spending_by_category's
# actual parameter type so beta_tool's schema generation emits a real
# JSON-schema enum - same reasoning as AwardType in tool_filters.py.
# TestLiteralTypesMatchVocabulary (tests/test_agent.py) asserts this can't
# silently drift from VALID_CATEGORIES.
Category = Literal[
    "awarding_agency", "awarding_subagency", "cfda", "country", "county",
    "defc", "district", "federal_account", "funding_agency",
    "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
    "state_territory",
]


def _normalize_category(category: str) -> str:
    """Same normalize-then-validate pattern as _normalize_award_type/
    _normalize_state: raises a clean, actionable USASpendingAPIError
    instead of letting an unrecognized category reach the live API and
    come back as a bare 404 with no guidance on what would have worked."""
    normalized = category.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in VALID_CATEGORIES:
        raise USASpendingAPIError(
            f"Unknown category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
        )
    return normalized


@traceable(run_type="tool", name="get_spending_by_category_raw")
def get_spending_by_category_raw(
    category: Category,
    agency_name: str | None,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SpendingByCategoryResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure — the @beta_tool wrapper decides how to
    present that to the model; this function stays presentation-free so the
    structured result is also available for chart-building later.

    category is validated against VALID_CATEGORIES before ever reaching the
    live API - same "code owns the exact vocabulary" pattern as award_type.

    agency_name is optional (2026-09-08) - see _build_filters' docstring
    for why (a cross-agency, recipient-only question has no answer at all
    if an agency must always be named), and for why recipient_id (an
    exact, precise filter - unlike recipient_name's text match) is only
    ever wired through here and on get_spending_over_time, never
    search_awards.

    Filter resolution (agency, award_type, recipient, amount, location,
    keywords, date_type, scope, naics/psc/cfda code) is delegated to
    _build_filters - see its docstring for the "no behavior change when
    the new params are omitted" guarantee and each failure mode.
    """
    category = _normalize_category(category)
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    return client.spending_by_category(category, filters, limit=limit)


@beta_tool
def get_spending_by_category(
    category: Category,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    agency_name: str | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for a fiscal year range, scoped by an awarding agency and/or a recipient, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions.

    At least one of agency_name, recipient_name, or recipient_id must be given - a query
    scoped by none of them would mean all federal spending, ever, which this tool refuses
    rather than silently running.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient, recipient_duns, state_territory. Enforced in code - any other value (including ones the API's own docs list, like object_class or tas, which 404 in practice) fails cleanly with this exact list rather than reaching the live API. recipient and recipient_duns return the same results for every case tested - either works for "top recipients" questions.
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        limit: Max number of results to return (default 5).
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
            Omit for a cross-agency question about one recipient (e.g. "which agencies has X
            received money from") - but then recipient_name or recipient_id must be set instead.
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos". This is an approximate text match - confirmed
            live that it can both miss real subsidiaries whose legal name differs from the
            parent company's, and sweep in unrelated similarly-named entities (e.g. an
            unrelated joint venture). If you already have a specific recipient_id (e.g. from
            search_recipients), pass that instead for an exact, correct total.
        recipient_id: Optional. The exact recipient_id from a prior search_recipients or
            get_recipient_details call (e.g. "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P") - an
            exact identifier, not a text match. Confirmed live to reproduce a recipient's true
            total precisely. Prefer this over recipient_name whenever you have it. Do not
            guess or construct a recipient_id.
        min_amount: Optional. Restrict to spending of at least this dollar amount.
        max_amount: Optional. Restrict to spending of at most this dollar amount.
        performed_in_state: Optional. Restrict to spending on work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to spending on recipients
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different totals.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default: any transaction/modification in the window - note this means
            a multi-year award active in more than one fiscal year appears in results for EACH
            of those years), date_signed (the award's original signing date), last_modified_date,
            or new_awards_only (only awards that originated in this window - use this for "what
            NEW contracts did X award in FY2024" as opposed to "what was X spending on in FY2024").
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511" for
            Custom Computer Programming Services. Must be the real code (2-6 digits) - if you
            only have a description, use category="naics" to browse the actual breakdown instead
            of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030"
            for Information Technology Software. Same guidance as naics_code: use category="psc"
            to browse if you don't have the exact code.
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    # Clamped here, at the model-facing boundary, not inside
    # get_spending_by_category_raw - the _raw function is the trusted
    # internal API (dev_tools scripts, tests can legitimately want an
    # uncapped value); this wrapper is the untrusted boundary a model's
    # tool call actually crosses, which is where the real abuse surface is.
    limit = _clamp_limit(limit)
    scope = _scope_label(agency_name, recipient_name, recipient_id)
    try:
        response = get_spending_by_category_raw(
            category,
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            limit,
            award_type=award_type,
            recipient_name=recipient_name,
            recipient_id=recipient_id,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_category failed for %s/%s: %s", scope, category, e)
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    context = _record_optional_filter_context(
        {
            "category": category,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
        },
        agency_name=agency_name,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    _record_tool_call("get_spending_by_category", response, context)

    if not response.results:
        return f"No {category} spending data found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
    return _wrap_untrusted("\n".join(lines) + note)


# Per spending_over_time.md's `group` enum - previously unvalidated in
# code at all (BACKLOG.md judged this "safe in practice" since the live
# API's own 400 for a bad value is already clean and informative, unlike
# category's bare 404). Added now anyway for consistency now that every
# other fixed-vocabulary param gets both a schema-level Literal and a
# runtime check - group was the one exception, not because it needed to
# stay one.
VALID_GROUPS = {"fiscal_year", "calendar_year", "quarter", "month"}

Group = Literal["fiscal_year", "calendar_year", "quarter", "month"]


def _normalize_group(group: str) -> str:
    normalized = group.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in VALID_GROUPS:
        raise USASpendingAPIError(
            f"Unknown group '{group}'. Must be one of: {', '.join(sorted(VALID_GROUPS))}"
        )
    return normalized


@traceable(run_type="tool", name="get_spending_over_time_raw")
def get_spending_over_time_raw(
    agency_name: str | None,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SpendingOverTimeResponse:
    """Call the API once, return the structured response. Same filter
    resolution (via _build_filters) as get_spending_by_category_raw -
    including agency_name's optionality and recipient_id's exactness, see
    that function's docstring."""
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    return client.spending_over_time(filters, group=_normalize_group(group))


@beta_tool
def get_spending_over_time(
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    agency_name: str | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Get USASpending spending trends over time for a fiscal year range, scoped by an awarding agency and/or a recipient, grouped by period. Use this for "how has X's spending changed/trended over time" questions.

    At least one of agency_name, recipient_name, or recipient_id must be given - a query
    scoped by none of them would mean all federal spending, ever, which this tool refuses
    rather than silently running.

    Args:
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        group: One of: fiscal_year, calendar_year, quarter, month. Default fiscal_year.
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
            Omit for a cross-agency trend for one recipient - but then recipient_name or
            recipient_id must be set instead.
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos". This is an approximate text match - confirmed
            live that it can both miss real subsidiaries whose legal name differs from the
            parent company's, and sweep in unrelated similarly-named entities (e.g. an
            unrelated joint venture). If you already have a specific recipient_id (e.g. from
            search_recipients), pass that instead for an exact, correct trend.
        recipient_id: Optional. The exact recipient_id from a prior search_recipients or
            get_recipient_details call (e.g. "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P") - an
            exact identifier, not a text match. Prefer this over recipient_name whenever you
            have it. Do not guess or construct a recipient_id.
        min_amount: Optional. Restrict to spending of at least this dollar amount.
        max_amount: Optional. Restrict to spending of at most this dollar amount.
        performed_in_state: Optional. Restrict to spending on work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to spending on recipients
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different totals.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default: any transaction/modification in the window - note this means
            a multi-year award active in more than one fiscal year contributes to EACH of those
            years' totals), date_signed (the award's original signing date), last_modified_date,
            or new_awards_only (only awards that originated in this window).
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511". Must be
            the real code - if you only have a description, use get_spending_by_category with
            category="naics" to browse the actual breakdown instead of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    scope = _scope_label(agency_name, recipient_name, recipient_id)
    try:
        response = get_spending_over_time_raw(
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            group,
            award_type=award_type,
            recipient_name=recipient_name,
            recipient_id=recipient_id,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_over_time failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "group": group,
        },
        agency_name=agency_name,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    _record_tool_call("get_spending_over_time", response, context)

    if not response.results:
        return f"No spending-over-time data found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [
        f"{_format_time_period(r.time_period)}: ${r.aggregated_amount:,.2f}"
        for r in response.results
    ]
    return _wrap_untrusted("\n".join(lines) + _format_api_messages(response.messages))


@traceable(run_type="tool", name="search_awards_raw")
def search_awards_raw(
    agency_name: str | None,
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SearchAwardsResponse:
    """Call the API once, return the structured response (results +
    page_metadata), sorted largest-amount-first (Award Amount, or Loan
    Value for loan award types - see _amount_field_for_award_type). This
    replaces the prior default order, which was verified live to be
    essentially arbitrary: an unsorted "top 5" NSF FY2023 contracts query
    returned awards from $7K to $7.2M while the true largest that year was
    $3.13B and never appeared. Unlike the filter params, which are all
    optional and behavior-preserving when omitted, this sort change is
    NOT optional - the old default had no meaningful ordering to
    preserve, so there's nothing to regress.

    Filter resolution (agency, award_type, recipient, amount, location) is
    delegated to _build_filters, same as the other two spending tools -
    including agency_name's optionality (2026-09-08). No recipient_id
    param here, unlike the other two: confirmed live that this endpoint
    silently ignores that filter entirely (the live API's own `messages`
    field says so explicitly) - recipient_name (an approximate text
    match) is the only recipient-scoping option this specific tool has.
    """
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    amount_field = _amount_field_for_award_type(award_type)
    fields = SEARCH_AWARDS_FIELDS_BASE + [amount_field]
    return client.search_awards(filters, fields=fields, limit=limit, sort=amount_field, order="desc")


@beta_tool
def search_awards(
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    agency_name: str | None = None,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Search for individual award records (specific contracts, grants, or loans) for a fiscal year range, scoped by an awarding agency and/or a recipient. Use this for "show me awards/contracts/grants from X" or "who received money from X" questions — as opposed to an aggregate breakdown or trend, which get_spending_by_category / get_spending_over_time answer instead. Results are ranked largest-amount-first by default — use this directly for "biggest"/"top N" questions.

    At least one of agency_name or recipient_name must be given - a query scoped
    by neither would mean all federal awards, ever, which this tool refuses rather
    than silently running.

    IMPORTANT about fiscal-year scoping: by default (date_type omitted), an
    award appears here if it had ANY transaction/modification in the queried
    fiscal year - not "this dollar amount was specifically obligated in this
    year." Award Amount/Loan Value is each award's current TOTAL value, not a
    per-year figure. A multi-year award active in more than one fiscal year
    will appear in results for EACH of those years showing the SAME total -
    do not sum or compare these across multiple fiscal-year calls to this
    tool as if they were period-scoped and additive; use
    get_spending_over_time for genuinely period-scoped, non-duplicative
    totals instead. If the question is really "what NEW contracts did X
    award in FY2024" rather than "what was X active on," set
    date_type="new_awards_only" instead of leaving this default.

    Each result also includes an internal_id - a separate, longer identifier
    from the plain Award ID (PIID/FAIN) shown alongside it. It isn't
    meaningful to a user, so don't quote it in your answer - pass it to
    get_award_details for full details on that specific award.

    Args:
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        award_type: The broad buckets are contracts, grants, loans (default contracts) - use one
            of these for a general "show me X's contracts/grants" question. For a question asking
            about a SPECIFIC sub-type rather than the broad category, use the specific value
            instead of guessing which broad bucket it falls under: bpa_call, purchase_order,
            delivery_order, definitive_contract (contract sub-types); direct_loan, guaranteed_loan
            (loan sub-types); block_grant, formula_grant, project_grant, cooperative_agreement
            (grant sub-types - e.g. "cooperative agreement" is cooperative_agreement, NOT
            contracts); insurance, other_financial_assistance, direct_payment_specified,
            direct_payment_unrestricted (other assistance types). Case/spacing/hyphens don't
            matter (e.g. "Cooperative Agreement" also works).
        limit: Max number of results to return (default 5).
        agency_name: Optional. The awarding agency's name, e.g. "National Science
            Foundation". Omit for a cross-agency question about one recipient - but then
            recipient_name must be set instead.
        recipient_name: Optional. Restrict to awards whose recipient name contains this
            text, e.g. "Leidos". This is an approximate text match, the only recipient
            filter this tool supports (unlike get_spending_by_category/get_spending_over_time,
            this tool does not accept a recipient_id - confirmed live it's silently
            ignored here) - confirmed live it can both miss real subsidiaries whose legal
            name differs from the parent company's, and sweep in unrelated similarly-named
            entities (e.g. an unrelated joint venture).
        min_amount: Optional. Restrict to awards worth at least this dollar amount.
        max_amount: Optional. Restrict to awards worth at most this dollar amount.
        performed_in_state: Optional. Restrict to awards for work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to awards whose recipient is
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different results.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. See the IMPORTANT note above - one of action_date (default),
            date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511". Must be
            the real code - if you only have a description, browse via get_spending_by_category
            (category="naics") instead of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    # Wrapper-level, not inside search_awards_raw - see the identical
    # comment on get_spending_by_category's clamp for why.
    limit = _clamp_limit(limit)
    scope = _scope_label(agency_name, recipient_name, None)
    try:
        results = search_awards_raw(
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            award_type,
            limit,
            recipient_name=recipient_name,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("search_awards failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "award_type": award_type,
        },
        agency_name=agency_name,
        recipient_name=recipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
        recipient_in_state=recipient_in_state,
    )
    _record_tool_call("search_awards", results, context)

    if not results.results:
        return f"No {award_type} awards found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    amount_field = _amount_field_for_award_type(award_type)
    lines = []
    for r in results.results:
        award_id = r.get("Award ID", "unknown")
        internal_id = r.get("generated_internal_id", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        amount = r.get(amount_field)
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        lines.append(f"{award_id} — {recipient}: {amount_str} [internal_id: {internal_id}]")
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results)) + _format_api_messages(results.messages)
    return _wrap_untrusted("\n".join(lines) + note)


@traceable(run_type="tool", name="get_award_details_raw")
def get_award_details_raw(award_id: str) -> dict[str, Any]:
    """Call the API once, return the raw response dict. Raises
    USASpendingAPIError if award_id doesn't resolve - most commonly
    because a plain PIID/FAIN was passed instead of the hash-style
    internal_id search_awards's own output now carries alongside it (see
    client.get_award's docstring)."""
    client = _get_usaspending_client()
    return client.get_award(award_id)


@traceable(run_type="tool", name="get_idv_amounts_raw")
def get_idv_amounts_raw(award_id: str) -> IDVAmountsResponse:
    """Call the API once, return the structured child/grandchild-order
    rollup. Raises USASpendingAPIError on failure - see
    IDVAmountsResponse's docstring for what this covers and why it's a
    separate call from get_award_details_raw."""
    client = _get_usaspending_client()
    return client.get_idv_amounts(award_id)


def _agency_label(agency: dict[str, Any] | None) -> str:
    """awarding_agency/funding_agency are Agency objects (id,
    has_agency_page, toptier_agency, subtier_agency, office_agency_name) -
    this pulls just the toptier name/abbreviation, the same trim applied
    to every other nested object here (recipient/location full addresses,
    executive compensation, etc. are all left out entirely - see the
    award-detail design walkthrough)."""
    if not agency:
        return "N/A"
    toptier = agency.get("toptier_agency") or {}
    name = toptier.get("name")
    abbreviation = toptier.get("abbreviation")
    if name and abbreviation:
        return f"{name} ({abbreviation})"
    return name or abbreviation or "N/A"


def _location_label(location: dict[str, Any] | None, *, full: bool = True) -> str:
    """full=False shows state only - used for record_type 1/3 recipients
    (aggregate/PII-redacted), where the live API still returns city/county/
    zip-level detail even though the recipient's own identity is withheld
    (confirmed live 2026-09-08: a PII-redacted individual recipient came
    back with city/county/zip populated) - showing that full granularity
    here would undercut the redaction's own purpose, so this app
    deliberately narrows it to state-level for those two cases rather
    than forwarding whatever the API happens to return."""
    if not location:
        return "N/A"
    state = location.get("state_name") or location.get("state_code")
    if not full:
        return state or "N/A"
    city = location.get("city_name")
    if city and state:
        return f"{city}, {state}"
    return state or location.get("country_name") or "N/A"


def _format_period_of_performance(pop: dict[str, Any] | None) -> str | None:
    """None (not a placeholder string) when neither date is present - found
    live 2026-09-08 on a real SBA-guaranteed-loan record where both
    start_date/end_date were null; the prior "? to ?" fallback rendered as
    noise instead of just omitting the line, so callers skip the line
    entirely on None rather than printing a fallback."""
    if not pop or (not pop.get("start_date") and not pop.get("end_date")):
        return None
    start = pop.get("start_date") or "?"
    end = pop.get("end_date") or "?"
    label = f"{start} to {end}"
    potential = pop.get("potential_end_date")
    if potential and potential != end:
        label += f" (potential end date {potential})"
    return label


def _format_contract_or_idv(
    data: dict[str, Any], child_order_rollup: IDVAmountsResponse | None = None
) -> str:
    """Shared formatting for ContractResponse and IDVResponse - identical
    field shape per award_id.md (IDVResponse just adds a nullable
    total_outlay this app doesn't surface). See the award-detail design
    walkthrough for which of the ~35 top-level fields (plus the ~60 on
    latest_transaction_contract_data) made the cut: money, dates, parties,
    parent-award linkage, and 5 procurement-detail fields an analyst
    actually asks about - not the ~55 FAR policy-code fields, not
    executive compensation, not the Treasury-account-level totals
    (total_account_obligation/outlay, the *_by_defc arrays) that come from
    a different, separately-timed DATA Act submission (File C) than this
    award's own total_obligation (File D2) - see the File C/D linkage
    finding in the design discussion for why those two are never mixed
    into one answer.

    child_order_rollup is only ever passed for category == "idv" (get_award_details
    only fetches it when both include_child_orders is set and the award is an
    IDV) - it replaces the generic "$0 doesn't mean nothing happened" caveat
    with the real child/grandchild-order totals from GET /api/v2/idvs/amounts/
    {award_id}/."""
    lines = [f"{data.get('type_description', 'Unknown type')} ({data.get('piid', 'unknown PIID')})"]
    if data.get("description"):
        lines.append(f"Description: {data['description']}")
    total_obligation = data.get("total_obligation") or 0
    ceiling = data.get("base_and_all_options") or 0
    lines.append(f"Total obligated: ${total_obligation:,.2f}, ceiling (base + all options): ${ceiling:,.2f}")
    if data.get("date_signed"):
        lines.append(f"Date signed: {data['date_signed']}")
    pop_label = _format_period_of_performance(data.get('period_of_performance'))
    if pop_label:
        lines.append(f"Period of performance: {pop_label}")
    lines.append(f"Awarding agency: {_agency_label(data.get('awarding_agency'))}")
    if data.get("funding_agency"):
        lines.append(f"Funding agency: {_agency_label(data['funding_agency'])}")

    recipient = data.get("recipient") or {}
    lines.append(
        f"Recipient: {recipient.get('recipient_name', 'unknown')} ({_location_label(recipient.get('location'))})"
    )
    lines.append(f"Place of performance: {_location_label(data.get('place_of_performance'))}")

    subaward_count = data.get("subaward_count") or 0
    if subaward_count:
        total_sub = data.get("total_subaward_amount")
        suffix = f", totaling ${total_sub:,.2f}" if total_sub else ""
        lines.append(f"Subawards: {subaward_count}{suffix}")

    contract_details = data.get("latest_transaction_contract_data") or {}
    if contract_details.get("extent_competed_description"):
        offers = contract_details.get("number_of_offers_received")
        suffix = f" ({offers} offers received)" if offers else ""
        lines.append(f"Competition: {contract_details['extent_competed_description']}{suffix}")
    if contract_details.get("type_of_contract_pricing_description"):
        lines.append(f"Contract pricing: {contract_details['type_of_contract_pricing_description']}")
    if contract_details.get("naics_description"):
        lines.append(f"NAICS: {contract_details['naics_description']}")
    if contract_details.get("product_or_service_description"):
        lines.append(f"Product/service: {contract_details['product_or_service_description']}")

    parent = data.get("parent_award")
    if parent:
        # generated_unique_award_id is the parent IDV's own internal_id -
        # found missing live 2026-09-08: asked a real "how much has been
        # ordered under this vehicle" question, and the model could only
        # reach a CHILD delivery order (this contract itself), never the
        # parent IDV, because this line hadn't been surfacing the one
        # value a follow-up get_award_details(include_child_orders=True)
        # call on the vehicle itself actually needs.
        vehicle_type = parent.get("type_of_idc_description") or parent.get("idv_type_description") or "contract vehicle"
        parent_internal_id = parent.get("generated_unique_award_id", "unknown")
        lines.append(
            f"Issued under parent IDV {parent.get('piid', 'unknown')} "
            f"({parent.get('agency_name', 'unknown agency')}, {vehicle_type}) "
            f"[internal_id: {parent_internal_id}]"
        )

    if data.get("category") == "idv":
        if child_order_rollup is not None:
            lines.append(
                f"Orders placed against this vehicle: {child_order_rollup.child_award_count} child awards "
                f"totaling ${child_order_rollup.child_award_total_obligation:,.2f} "
                f"(ceiling ${child_order_rollup.child_award_base_and_all_options_value:,.2f})"
            )
            if child_order_rollup.grandchild_award_count:
                lines.append(
                    f"Plus {child_order_rollup.grandchild_award_count} grandchild orders (orders placed "
                    f"against child IDVs nested under this one) totaling "
                    f"${child_order_rollup.grandchild_award_total_obligation:,.2f}"
                )
        else:
            lines.append(
                "(Note: this is a contract vehicle (IDV), not itself a spending transaction - the total "
                "obligated above reflects only the vehicle's own direct activity, not the orders placed "
                "against it. A real IDV can show $0 here while still being an active, heavily-used "
                "vehicle - do not present this figure as the total spent under this contract. Call "
                "again with include_child_orders=True for the real child-order rollup.)"
            )
    return "\n".join(lines)


# record_type 1 (Aggregate Record) and 3 (Non-Aggregate Record to an
# Individual Recipient with Redacted PII) per this project's own ingested
# Glossary ("Record Type" entry) - confirmed live 2026-09-08 against real
# examples of both: record_type 1 shows recipient_name literally
# "MULTIPLE RECIPIENTS", record_type 3 shows "REDACTED DUE TO PII". Never
# shown to the model/user as the raw sentinel string - both get an
# explanatory label instead, and both get the state-only location (see
# _location_label's docstring for why full location isn't shown here).
_AGGREGATE_RECIPIENT_LABELS = {
    1: "Multiple recipients (aggregate award - individual identities aren't published, to protect personal privacy)",
    3: "Individual recipient (redacted - not published, to protect personal privacy)",
}


def _format_financial_assistance(data: dict[str, Any]) -> str:
    """Formatting for FinancialAssistanceResponse (grants/loans/direct
    payments/other assistance) - a genuinely different shape from
    ContractResponse/IDVResponse (fain/uri instead of piid, cfda_info
    instead of NAICS/PSC, no competition data, no parent-award linkage).

    total_subsidy_cost/total_loan_value and non_federal_funding/
    total_funding are shown only for their matching category (loans,
    grant) - conditioned on category, not on nullness, since live
    verification (2026-09-08) found these come back 0.0, not null, on
    every record they don't apply to (the award_id.md contract's own
    "null except for X" claim doesn't hold in practice).

    transaction_obligated_amount is deliberately never shown - live
    verification traced it to a File C (Treasury-account-level financial
    reporting) sum, a genuinely different, separately-timed DATA Act
    submission from total_obligation's File D2 (award/transaction) source
    (usaspending-api's own C_to_D_Linkage.md documents the two are
    reconciled by best-effort matching, not guaranteed to agree) - same
    "different pipeline, will confuse if shown unlabeled" reasoning as
    excluding total_account_obligation/outlay for contracts/IDVs above."""
    award_number = data.get("fain") or data.get("uri") or "unknown"
    lines = [f"{data.get('type_description', 'Unknown type')} ({award_number})"]
    if data.get("description"):
        lines.append(f"Description: {data['description']}")
    total_obligation = data.get("total_obligation") or 0
    category = data.get("category")
    # For a loan, total_obligation came back $0.00 on a real live example
    # ($98.4M SBA-guaranteed loan) - a bare "$0.00" line right above "Loan
    # value: $98,400,000.00" reads as contradictory even if it's a
    # correct value for whatever total_obligation specifically measures
    # here (not independently confirmed - the live example only shows
    # both total_obligation and total_subsidy_cost at $0, which doesn't
    # prove they're the same figure). Suppressing it when zero avoids
    # asserting a semantic claim that isn't verified, while still not
    # hiding a genuinely nonzero value if one shows up on some other loan.
    if not (category == "loans" and total_obligation == 0):
        lines.append(f"Total obligated: ${total_obligation:,.2f}")

    if category == "loans" and data.get("total_loan_value"):
        subsidy = data.get("total_subsidy_cost")
        suffix = f", subsidy cost ${subsidy:,.2f}" if subsidy else ""
        lines.append(f"Loan value: ${data['total_loan_value']:,.2f}{suffix}")
    elif category == "grant" and data.get("total_funding"):
        non_federal = data.get("non_federal_funding")
        suffix = f" (of which ${non_federal:,.2f} non-federal)" if non_federal else ""
        lines.append(f"Total funding: ${data['total_funding']:,.2f}{suffix}")

    if data.get("date_signed"):
        lines.append(f"Date signed: {data['date_signed']}")
    pop_label = _format_period_of_performance(data.get('period_of_performance'))
    if pop_label:
        lines.append(f"Period of performance: {pop_label}")
    lines.append(f"Awarding agency: {_agency_label(data.get('awarding_agency'))}")
    if data.get("funding_agency"):
        lines.append(f"Funding agency: {_agency_label(data['funding_agency'])}")

    recipient = data.get("recipient") or {}
    record_type = data.get("record_type")
    if record_type in _AGGREGATE_RECIPIENT_LABELS:
        location = _location_label(recipient.get("location"), full=False)
        lines.append(f"Recipient: {_AGGREGATE_RECIPIENT_LABELS[record_type]} ({location})")
    else:
        lines.append(
            f"Recipient: {recipient.get('recipient_name', 'unknown')} ({_location_label(recipient.get('location'))})"
        )

    lines.append(f"Place of performance: {_location_label(data.get('place_of_performance'))}")

    subaward_count = data.get("subaward_count") or 0
    if subaward_count:
        total_sub = data.get("total_subaward_amount")
        suffix = f", totaling ${total_sub:,.2f}" if total_sub else ""
        lines.append(f"Subawards: {subaward_count}{suffix}")

    cfda_info = data.get("cfda_info") or []
    if cfda_info:
        programs = "; ".join(f"{c.get('cfda_number', '?')} {c.get('cfda_title') or ''}".strip() for c in cfda_info)
        lines.append(f"Assistance Listing(s): {programs}")

    return "\n".join(lines)


def _format_award_details(
    data: dict[str, Any], child_order_rollup: IDVAmountsResponse | None = None
) -> str:
    if data.get("category") in ("contract", "idv"):
        return _format_contract_or_idv(data, child_order_rollup)
    return _format_financial_assistance(data)


@beta_tool
def get_award_details(award_id: str, include_child_orders: bool = False) -> str:
    """Get full details about one specific award: a single contract, IDV (contract vehicle), grant, loan, or other financial assistance record. Returns description, dates, competition data (for contracts), recipient, funding breakdown, and (for contracts issued under an IDV) parent-vehicle info. Use this for "tell me more about this award/contract/grant" follow-up questions after search_awards — not for browsing or listing multiple awards, which search_awards already does.

    Args:
        award_id: The internal_id value shown alongside a search_awards result (e.g.
            "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-") — NOT the plain Award ID/PIID/FAIN
            shown next to it, which this endpoint doesn't accept. Only call this with an
            internal_id you already have — either from a prior search_awards result, or from a
            prior get_award_details call's own "Issued under parent IDV ... [internal_id: ...]"
            line, if the question is about the parent VEHICLE rather than the specific contract
            found by search_awards (e.g. "how much has been ordered under this IDV" — call
            get_award_details again on the parent's internal_id, with include_child_orders=True,
            rather than answering from the child contract's own total). Do not guess or
            construct an internal_id.
        include_child_orders: Set True only when the award is an IDV (a contract vehicle — BPA,
            GWAC, or multi-award IDC) AND the question is specifically about how much has been
            ordered under it (e.g. "how much has actually been spent under this contract
            vehicle"). False by default — it costs a second live API call and is meaningless for
            a plain contract/grant/loan/etc. An IDV's own total obligated (shown above regardless
            of this flag) reflects only the vehicle's own direct activity, not the orders placed
            against it — a real, active IDV can show $0 there. Setting this True fetches the
            actual rollup: how many child orders (and, for a nested vehicle, grandchild orders)
            exist and what they total.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        data = get_award_details_raw(award_id)
    except USASpendingAPIError as e:
        logger.warning("get_award_details failed for %s: %s", award_id, e)
        return f"This query failed: {e}."

    child_order_rollup = None
    if include_child_orders and data.get("category") == "idv":
        try:
            child_order_rollup = get_idv_amounts_raw(award_id)
        except USASpendingAPIError as e:
            # Degrade gracefully rather than failing the whole call over an
            # enhancement fetch - _format_contract_or_idv falls back to its
            # existing caveat when child_order_rollup is None.
            logger.warning("get_idv_amounts failed for %s: %s", award_id, e)

    _record_tool_call(
        "get_award_details",
        data,
        {"award_id": award_id, "piid": data.get("piid") or data.get("fain") or data.get("uri")},
    )
    return _wrap_untrusted(_format_award_details(data, child_order_rollup))


def _format_recipient_level(level: str) -> str:
    return {"P": "parent", "C": "child", "R": "standalone"}.get(level, level)


def _format_recipient_listing(listing: RecipientListing) -> str:
    """One line per search_recipients candidate. amount is always
    trailing-12-months (RecipientListing.amount's own docstring) - labeled
    explicitly so it's never mistaken for the all-time total
    get_recipient_details can give instead."""
    ids = [f"UEI {listing.uei}" if listing.uei else None, f"DUNS {listing.duns}" if listing.duns else None]
    id_str = f" ({', '.join(i for i in ids if i)})" if any(ids) else ""
    return (
        f"{listing.name or 'unknown'} [{_format_recipient_level(listing.recipient_level)}]{id_str} - "
        f"${listing.amount:,.2f} (last 12 months) [recipient_id: {listing.id}]"
    )


def _format_recipient_address(location: RecipientLocation | None) -> str:
    """Full street address - the real live usaspending.gov recipient page
    itself shows this in full for a normal business (confirmed by pasting
    the real Boeing page in) - unlike the award-side recipient/place-of-
    performance trim (state/city only), which exists for a different
    reason (redacting an individual's home address). Callers use
    _format_recipient_state_only instead of this specifically for the
    redacted/aggregate bucket case - see _format_recipient_overview."""
    if not location:
        return "N/A"
    street = ", ".join(p for p in [location.address_line1, location.address_line2, location.address_line3] if p)
    city_state_zip = " ".join(p for p in [location.city_name, location.state_code, location.zip] if p)
    parts = [p for p in [street, city_state_zip, location.country_name] if p]
    return ", ".join(parts) if parts else "N/A"


def _format_recipient_state_only(location: RecipientLocation | None) -> str:
    if not location:
        return "N/A"
    return location.state_code or location.country_name or "N/A"


# The live sentinel string for a pooled bucket of PII-redacted individual
# recipients (RecipientOverview has no typed flag for this, unlike the
# award side's record_type - only this literal name string) - confirmed
# live 2026-09-08 against a real example: $14.9B, 2.24M transactions,
# clearly not one person. Never shown verbatim to the model/user as if it
# were a real name.
_REDACTED_RECIPIENT_NAME = "REDACTED DUE TO PII"


def _format_recipient_overview(overview: RecipientOverview) -> str:
    is_redacted = overview.name == _REDACTED_RECIPIENT_NAME
    lines: list[str] = []

    if is_redacted:
        lines.append(
            "This recipient_id represents a pooled aggregate of many PII-redacted individual "
            "recipients, not one person or entity - the totals below are NOT one recipient's "
            "spending. (Real example confirmed live: $14.9B across 2.24M transactions.)"
        )
    else:
        lines.append(overview.name or "unknown")
        if overview.alternate_names:
            lines.append(f"Also known as: {', '.join(overview.alternate_names)}")

    lines.append(f"Recipient level: {_format_recipient_level(overview.recipient_level)}")
    if overview.uei:
        lines.append(f"UEI: {overview.uei}")
    if overview.duns:
        lines.append(f"Legacy DUNS: {overview.duns}")

    if overview.parent_id and overview.parent_id != overview.recipient_id:
        lines.append(f"Parent: {overview.parent_name or 'unknown'} [recipient_id: {overview.parent_id}]")

    location_label = _format_recipient_state_only(overview.location) if is_redacted else _format_recipient_address(overview.location)
    lines.append(f"Location: {location_label}")

    if overview.business_types:
        readable = ", ".join(bt.replace("_", " ").title() for bt in overview.business_types)
        lines.append(f"Business types: {readable}")

    lines.append(
        f"Total: ${overview.total_transaction_amount:,.2f} across {overview.total_transactions:,} transactions"
    )
    # Always shown, not suppressed at zero, unlike get_award_details's loan
    # case - the real live usaspending.gov page itself always shows this
    # line ("$0 from 0 transactions"), confirmed by pasting the real
    # Boeing page in, and there's no adjacent nonzero figure here to make
    # a zero read as contradictory the way it did for a guaranteed loan.
    lines.append(
        f"Face value of loans: ${overview.total_face_value_loan_amount:,.2f} across "
        f"{overview.total_face_value_loan_transactions:,} transactions"
    )

    return "\n".join(lines)


@beta_tool
def search_recipients(keyword: str, award_type: RecipientAwardType = "all", limit: int = 10) -> str:
    """Search for a recipient (company, organization, or individual) by name, UEI, or DUNS number, to find its exact recipient_id for a precise follow-up query (get_recipient_details, or the recipient_id parameter on get_spending_by_category/get_spending_over_time). Use this whenever a question names a specific real recipient — do not guess a recipient_id, and prefer this over a bare recipient_name text filter whenever precision matters.

    A plain company name is genuinely ambiguous at this scale — confirmed live that "Leidos" and "Boeing" each resolve to 6+ distinct recipient_ids sharing the exact same display name (parent companies, subsidiaries, and historical registrations from mergers/acquisitions). This tool shows every real candidate rather than silently picking one. If several results share a name, prefer the one with recipient level "parent" for a "how much has this company received in total" question — confirmed live to be a true, complete rollup across all of that company's own child registrations, to the penny. Ask the user to disambiguate if it's still unclear which candidate they mean.

    An exact UEI or DUNS as the keyword returns a single, precise match (confirmed live) — use one directly if you already have it.

    Args:
        keyword: A recipient's name, UEI, or DUNS number, e.g. "Boeing" or "NU2UC8MX6NK1".
        award_type: Optional. Restrict to one broad award-type bucket — all (default),
            contracts, grants, loans, direct_payments, or other_financial_assistance. A
            different, coarser vocabulary than every other tool's award_type parameter here —
            no sub-type granularity (no cooperative_agreement, no bpa_call).
        limit: Max number of candidates to return (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    limit = _clamp_limit(limit)
    try:
        normalized_award_type = _normalize_recipient_award_type(award_type)
        client = _get_usaspending_client()
        response = client.search_recipients(keyword, award_type=normalized_award_type, limit=limit)
    except USASpendingAPIError as e:
        logger.warning("search_recipients failed for %r: %s", keyword, e)
        return f"This query failed: {e}."

    _record_tool_call("search_recipients", response, {"keyword": keyword})

    if not response.results:
        return f"No recipients found matching '{keyword}'."

    lines = [_format_recipient_listing(r) for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
    return _wrap_untrusted("\n".join(lines) + note)


@beta_tool
def get_recipient_details(recipient_id: str, year: str = "all") -> str:
    """Get full profile details for one specific, already-resolved recipient: identity (name, alternate names, UEI/Legacy DUNS), parent relationship, address, business types, and total federal transaction amount for the given time period. Use this as a follow-up after search_recipients has resolved a specific recipient_id — not for browsing or searching by name, which search_recipients already does.

    year defaults to "all" (the recipient's entire history), not the live API's own default
    of "latest" (trailing 12 months) — "latest" would just repeat the same number
    search_recipients already showed for the same candidate (confirmed live: search_recipients's
    own amount is always trailing-12-months and never respects year), so defaulting here to
    "all" gives a genuinely different, additive answer instead of restating one.

    Args:
        recipient_id: The exact recipient_id from a prior search_recipients result (e.g.
            "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"). Do not guess or construct one.
        year: A specific fiscal year (e.g. "2023"), "all" (default — the recipient's entire
            history), or "latest" (trailing 12 months — the same window search_recipients
            already shows, so rarely useful here unless explicitly asked for).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        client = _get_usaspending_client()
        overview = client.get_recipient(recipient_id, year=year)
    except USASpendingAPIError as e:
        logger.warning("get_recipient_details failed for %s: %s", recipient_id, e)
        return f"This query failed: {e}."

    _record_tool_call("get_recipient_details", overview, {"recipient_id": recipient_id, "name": overview.name})
    return _wrap_untrusted(_format_recipient_overview(overview))

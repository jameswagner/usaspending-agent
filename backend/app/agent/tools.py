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
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    AgencyYearBudget,
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
    Scope,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
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


@beta_tool
def get_agency_budget(agency_name: str, start_fiscal_year: int, end_fiscal_year: int) -> str:
    """Get an agency's actual appropriated budgetary resources, obligations, and outlays for a fiscal year range. Use this specifically for "what is X's budget," "how much money does X have," or "how much has X actually paid out" questions.

    This is a genuinely different concept from what get_spending_by_category, get_spending_over_time, and search_awards report: those three track money obligated against specific contracts, grants, and loans (award-level spending activity), not the agency's appropriated budget authority. An agency's total budgetary resources for a fiscal year is NOT the same number as its total award spending in that year, and the two should never be presented as if interchangeable - if asked about budget/appropriations specifically, use this tool, not the spending tools, even though both involve dollar figures for the same agency.

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021. This endpoint's
            own data only goes back to FY2017 (a shorter history than the FY2008 floor the
            spending tools have) - a range starting earlier than that will just return whatever
            years are actually available, not error.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
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

    lines = [
        f"FY{y.fiscal_year}: budgetary resources {_fmt(y.agency_budgetary_resources)}, "
        f"obligated {_fmt(y.agency_total_obligated)}, outlayed {_fmt(y.agency_total_outlayed)}"
        for y in sorted(years, key=lambda y: y.fiscal_year)
    ]
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
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    award_type: AwardType | None = None,
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
) -> SpendingByCategoryResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure — the @beta_tool wrapper decides how to
    present that to the model; this function stays presentation-free so the
    structured result is also available for chart-building later.

    category is validated against VALID_CATEGORIES before ever reaching the
    live API - same "code owns the exact vocabulary" pattern as award_type.

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
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    award_type: AwardType | None = None,
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
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for one awarding agency and fiscal year range, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient, recipient_duns, state_territory. Enforced in code - any other value (including ones the API's own docs list, like object_class or tas, which 404 in practice) fails cleanly with this exact list rather than reaching the live API. recipient and recipient_duns return the same results for every case tested - either works for "top recipients" questions.
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        limit: Max number of results to return (default 5).
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos".
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
    try:
        response = get_spending_by_category_raw(
            category,
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            limit,
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
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_category failed for %s/%s: %s", agency_name, category, e)
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    context = _record_optional_filter_context(
        {
            "agency_name": agency_name,
            "category": category,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
        },
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
    _record_tool_call("get_spending_by_category", response, context)

    if not response.results:
        return f"No {category} spending data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
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
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    award_type: AwardType | None = None,
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
) -> SpendingOverTimeResponse:
    """Call the API once, return the structured response. Same filter
    resolution (via _build_filters) as get_spending_by_category_raw."""
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
    return client.spending_over_time(filters, group=_normalize_group(group))


@beta_tool
def get_spending_over_time(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    award_type: AwardType | None = None,
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
    """Get USASpending spending trends over time for one awarding agency, grouped by period. Use this for "how has X's spending changed/trended over time" questions.

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        group: One of: fiscal_year, calendar_year, quarter, month. Default fiscal_year.
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos".
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
    try:
        response = get_spending_over_time_raw(
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            group,
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
    except USASpendingAPIError as e:
        logger.warning("get_spending_over_time failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "agency_name": agency_name,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "group": group,
        },
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
    _record_tool_call("get_spending_over_time", response, context)

    if not response.results:
        return f"No spending-over-time data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [
        f"{_format_time_period(r.time_period)}: ${r.aggregated_amount:,.2f}"
        for r in response.results
    ]
    return _wrap_untrusted("\n".join(lines))


@traceable(run_type="tool", name="search_awards_raw")
def search_awards_raw(
    agency_name: str,
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
    delegated to _build_filters, same as the other two spending tools.
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
    agency_name: str,
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
) -> str:
    """Search for individual award records (specific contracts, grants, or loans) for one awarding agency and fiscal year range. Use this for "show me awards/contracts/grants from X" or "who received money from X" questions — as opposed to an aggregate breakdown or trend, which get_spending_by_category / get_spending_over_time answer instead. Results are ranked largest-amount-first by default — use this directly for "biggest"/"top N" questions.

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

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
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
        recipient_name: Optional. Restrict to awards whose recipient name contains this
            text, e.g. "Leidos".
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
        logger.warning("search_awards failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "agency_name": agency_name,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "award_type": award_type,
        },
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
        return f"No {award_type} awards found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    amount_field = _amount_field_for_award_type(award_type)
    lines = []
    for r in results.results:
        award_id = r.get("Award ID", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        amount = r.get(amount_field)
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        lines.append(f"{award_id} — {recipient}: {amount_str}")
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results))
    return _wrap_untrusted("\n".join(lines) + note)

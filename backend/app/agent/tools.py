"""The five @beta_tool-decorated functions the agent calls, and their _raw
variants (structured Pydantic responses, presentation-free) that also feed
should_chart/build_tool_citation after the tool-calling loop finishes.
"""
from __future__ import annotations

import contextvars

from anthropic import beta_tool

from backend.app.tools.usaspending_client import (
    AdvancedFilters,
    AgencyFilter,
    SpendingByCategoryResponse,
    SpendingOverTimeResponse,
    TimePeriod,
    USASpendingAPIError,
)

from .clients import (
    RERANK_CONFIDENCE_THRESHOLD,
    _get_retriever,
    _get_usaspending_client,
)
from .response_shaping import _format_time_period, fiscal_year_to_date_range

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
    """Search the Analyst's Guide to Federal Spending Data for conceptual or definitional information about USASpending — what a term means, how a data element is defined, which fields contain what.

    Args:
        query: What to search for in the guide.
    """
    results = _get_retriever().retrieve(query, top_k=3)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]
    if not matches:
        return "No relevant content found in the Analyst's Guide for this query."

    _record_tool_call("search_guide", matches)

    return "\n\n---\n\n".join(f"[Page {m['page_start']}]\n{m['text']}" for m in matches)


@beta_tool
def lookup_agency(name: str) -> str:
    """Look up a federal agency by name to get its basic profile: toptier code, abbreviation, mission, website, and subtier agency count. Use this for questions about what a specific agency is or does, or as a first step before any spending-data question that needs an agency's toptier code.

    Args:
        name: The agency name to search for, e.g. "National Science Foundation" or "NSF".
    """
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(name)
    if agency is None:
        return f"No agency found matching '{name}'."

    overview = client.get_agency_overview(agency.toptier_code)
    _record_tool_call("lookup_agency", overview, {"name": name})
    return (
        f"Agency: {overview.name} ({overview.abbreviation})\n"
        f"Toptier code: {overview.toptier_code}\n"
        f"Fiscal year: {overview.fiscal_year}\n"
        f"Subtier agency count: {overview.subtier_agency_count}\n"
        f"Mission: {overview.mission or 'N/A'}\n"
        f"Website: {overview.website or 'N/A'}"
    )


def get_spending_by_category_raw(
    category: str,
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
) -> SpendingByCategoryResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure — the @beta_tool wrapper decides how to
    present that to the model; this function stays presentation-free so the
    structured result is also available for chart-building later.

    Resolves agency_name through find_agency_by_name first, rather than
    passing whatever string the caller gave straight into the filter: the
    live API silently returns zero results for an unrecognized name (e.g.
    "NSF" instead of "National Science Foundation") instead of erroring, so
    passing the raw string through would risk a false "no data" answer.
    """
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")

    start_date, end_date = fiscal_year_to_date_range(start_fiscal_year, end_fiscal_year)
    filters = AdvancedFilters(
        agencies=[AgencyFilter(type="awarding", tier="toptier", name=agency.agency_name)],
        time_period=[TimePeriod(start_date=start_date, end_date=end_date)],
    )
    return client.spending_by_category(category, filters, limit=limit)


@beta_tool
def get_spending_by_category(
    category: str,
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
) -> str:
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for one awarding agency and fiscal year range, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient_duns, state_territory. (Only these are live-verified; other category names the API contract lists, like object_class or tas, 404 in practice.)
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        limit: Max number of results to return (default 5).
    """
    try:
        response = get_spending_by_category_raw(category, agency_name, start_fiscal_year, end_fiscal_year, limit)
    except USASpendingAPIError as e:
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    _record_tool_call(
        "get_spending_by_category",
        response,
        {
            "agency_name": agency_name,
            "category": category,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
        },
    )

    if not response.results:
        return f"No {category} spending data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    return "\n".join(lines)


def get_spending_over_time_raw(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: str = "fiscal_year",
) -> SpendingOverTimeResponse:
    """Call the API once, return the structured response. Same split
    rationale, and same agency_name resolution, as get_spending_by_category_raw."""
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")

    start_date, end_date = fiscal_year_to_date_range(start_fiscal_year, end_fiscal_year)
    filters = AdvancedFilters(
        agencies=[AgencyFilter(type="awarding", tier="toptier", name=agency.agency_name)],
        time_period=[TimePeriod(start_date=start_date, end_date=end_date)],
    )
    return client.spending_over_time(filters, group=group)


@beta_tool
def get_spending_over_time(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: str = "fiscal_year",
) -> str:
    """Get USASpending spending trends over time for one awarding agency, grouped by period. Use this for "how has X's spending changed/trended over time" questions.

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        group: One of: fiscal_year, calendar_year, quarter, month. Default fiscal_year.
    """
    try:
        response = get_spending_over_time_raw(agency_name, start_fiscal_year, end_fiscal_year, group)
    except USASpendingAPIError as e:
        return f"This query failed: {e}."

    _record_tool_call(
        "get_spending_over_time",
        response,
        {
            "agency_name": agency_name,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "group": group,
        },
    )

    if not response.results:
        return f"No spending-over-time data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [
        f"{_format_time_period(r.time_period)}: ${r.aggregated_amount:,.2f}"
        for r in response.results
    ]
    return "\n".join(lines)


# award_type_codes has many more valid values than these three groups (see
# search_filters.md's Award Type section), but exposing the full code list
# to the model would mean a much larger error surface for little benefit -
# same reasoning as get_spending_by_category's constrained category list.
AWARD_TYPE_GROUPS: dict[str, list[str]] = {
    "contracts": ["A", "B", "C", "D"],
    "grants": ["02", "03", "04", "05"],
    "loans": ["07", "08"],
}

# A base field set valid across award types (per spending_by_award.md's
# "Base fields" list), so one fixed request shape works regardless of
# award_type - avoids the model needing to know which fields are only
# valid for contracts vs. loans vs. non-loan assistance.
SEARCH_AWARDS_FIELDS = [
    "Award ID",
    "Recipient Name",
    "Award Amount",
    "Awarding Agency",
    "Description",
]


def search_awards_raw(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: str = "contracts",
    limit: int = 5,
) -> list[dict]:
    """Call the API once, return the raw list of award result dicts. Same
    agency_name resolution as the other spending tools, for the same
    reason (an abbreviation would otherwise silently return zero results).
    """
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")

    award_type_codes = AWARD_TYPE_GROUPS.get(award_type)
    if award_type_codes is None:
        raise USASpendingAPIError(
            f"Unknown award_type '{award_type}'. Must be one of: {', '.join(AWARD_TYPE_GROUPS)}"
        )

    start_date, end_date = fiscal_year_to_date_range(start_fiscal_year, end_fiscal_year)
    filters = AdvancedFilters(
        agencies=[AgencyFilter(type="awarding", tier="toptier", name=agency.agency_name)],
        time_period=[TimePeriod(start_date=start_date, end_date=end_date)],
        award_type_codes=award_type_codes,
    )
    return client.search_awards(filters, fields=SEARCH_AWARDS_FIELDS, limit=limit)


@beta_tool
def search_awards(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: str = "contracts",
    limit: int = 5,
) -> str:
    """Search for individual award records (specific contracts, grants, or loans) for one awarding agency and fiscal year range. Use this for "show me awards/contracts/grants from X" or "who received money from X" questions — as opposed to an aggregate breakdown or trend, which get_spending_by_category / get_spending_over_time answer instead.

    Args:
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        award_type: One of: contracts, grants, loans. Default contracts.
        limit: Max number of results to return (default 5).
    """
    try:
        results = search_awards_raw(agency_name, start_fiscal_year, end_fiscal_year, award_type, limit)
    except USASpendingAPIError as e:
        return f"This query failed: {e}."

    _record_tool_call(
        "search_awards",
        results,
        {
            "agency_name": agency_name,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "award_type": award_type,
        },
    )

    if not results:
        return f"No {award_type} awards found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = []
    for r in results:
        award_id = r.get("Award ID", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        amount = r.get("Award Amount")
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        lines.append(f"{award_id} — {recipient}: {amount_str}")
    return "\n".join(lines)

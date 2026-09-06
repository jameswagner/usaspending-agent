"""Pure, unit-testable post-processing of already-returned tool results:
turning a structured tool result into a chart or a citation, and the
fiscal-year date math the live-data tools rely on. Deliberately has no
dependency on the Anthropic client or tool_runner - everything here operates
on plain data (structured results, context dicts) that's already been
returned from a tool call, not on how that call was made.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel


def current_fiscal_year(today: date | None = None) -> int:
    """The federal fiscal year in progress on `today` (defaults to the real
    current date). FY2021 = Oct 2020-Sep 2021, so Oct-Dec rolls over into
    the *next* calendar year's fiscal year.

    Exists so "most recent fiscal year" and similar relative phrases get
    resolved from the system clock in code, not guessed by the model from
    its training data - the same reasoning as fiscal_year_to_date_range
    below, applied to "which year is it" instead of "what are this year's
    date bounds." Demonstrated failure mode: asked for NSF's "most recent
    fiscal year" NAICS breakdown, the model answered FY2024 when FY2025
    data was already live - it has no way to know today's date unless told,
    so it fell back on a guess instead of computing one.

    `today` is a parameter (not read internally via date.today()) so tests
    can pin exact dates on either side of the Oct 1 boundary; callers that
    want "right now" just omit it. Must be called fresh per request, not
    cached at import/startup time, or a long-running server process will
    keep answering with the fiscal year it happened to boot in.
    """
    today = today or datetime.now(timezone.utc).date()
    return today.year + 1 if today.month >= 10 else today.year


def fiscal_year_to_date_range(start_fiscal_year: int, end_fiscal_year: int) -> tuple[str, str]:
    """Convert a fiscal year range to the API's YYYY-MM-DD date bounds.

    Federal fiscal years are named by the calendar year they END in - FY2021
    runs 2020-10-01 through 2021-09-30. Doing this conversion in code rather
    than asking the model to compute date strings directly closes off a
    demonstrated failure mode: asked for "2021 to 2024," the model passed
    start_date="2021-10-01" - the start of FY2022, not FY2021 - an off-by-
    one-fiscal-year error in date arithmetic the model has no reliable way
    to get right consistently. The model only has to identify which years
    are being asked about now, not compute a date boundary.
    """
    start_date = f"{start_fiscal_year - 1}-10-01"
    end_date = f"{end_fiscal_year}-09-30"
    return start_date, end_date


def _format_time_period(period) -> str:
    # Labeled "FY"/"CY" explicitly (not a bare year number) so the label
    # unambiguously carries fiscal-vs-calendar-year meaning through to
    # whatever prose the model writes from it, rather than depending on the
    # model to re-add that context itself.
    if period.fiscal_year:
        label = f"FY{period.fiscal_year}"
    elif period.calendar_year:
        label = f"CY{period.calendar_year}"
    else:
        label = "?"
    if period.quarter:
        label += f" Q{period.quarter}"
    if period.month:
        label += f" month {period.month}"
    return label


class ChartSpec(BaseModel):
    chart_type: Literal["bar", "line"]
    title: str
    labels: list[str]
    values: list[float]


class Citation(BaseModel):
    chunk_id: str
    source: str
    # Exactly one of these is set - page for the page-based Analyst's
    # Guide, term for the USASpending Glossary (which has no real page
    # number to point to). Two optional fields rather than a shared
    # required "locator" string so the frontend can render each source
    # type's natural format ("source, page N" vs "source: term").
    page: int | None = None
    term: str | None = None


class ToolCitation(BaseModel):
    """Citation for a live USASpending.gov API call - the live-data analog
    of Citation. There's no chunk id or page number for a live query, so
    this doesn't try to force that shape; instead it cites the tool and the
    exact parameters used, which is what a user or auditor actually needs
    to re-run the same query and verify the numbers themselves."""

    tool_name: str
    parameters: dict[str, str | int]
    description: str


# Tools whose results are never chart-worthy by shape (free text / a single
# profile), regardless of what's in the result.
NEVER_CHART_TOOLS = {"search_guide", "lookup_agency", "search_awards", "code_execution"}


def should_chart(tool_name: str, structured_result, context: dict | None = None) -> ChartSpec | None:
    """Deterministic, unit-testable chart-eligibility check keyed on the
    actual result's cardinality — not on guessing intent from the question,
    since the tool call has already resolved that ambiguity by the time
    we're deciding whether to chart. A single data point reads better as
    prose than a one-bar/one-point chart, so both branches require 2+
    results.

    context (e.g. {"agency_name": ...}) is optional and only used to make
    the title distinguish multiple charts of the same type in one turn
    (e.g. comparing two agencies' trends) - omitting it just yields a
    more generic title, not a failure.
    """
    if tool_name in NEVER_CHART_TOOLS:
        return None

    agency_name = (context or {}).get("agency_name")

    if tool_name == "get_spending_by_category":
        if len(structured_result.results) < 2:
            return None
        title = f"Spending by {structured_result.category}"
        if agency_name:
            title += f" — {agency_name}"
        return ChartSpec(
            chart_type="bar",
            title=title,
            labels=[r.name or r.code or "unknown" for r in structured_result.results],
            values=[r.amount for r in structured_result.results],
        )

    if tool_name == "get_spending_over_time":
        if len(structured_result.results) < 2:
            return None
        title = f"Spending over time ({structured_result.group})"
        if agency_name:
            title += f" — {agency_name}"
        return ChartSpec(
            chart_type="line",
            title=title,
            labels=[_format_time_period(r.time_period) for r in structured_result.results],
            values=[r.aggregated_amount for r in structured_result.results],
        )

    return None


def build_tool_citation(tool_name: str, context: dict) -> ToolCitation | None:
    """Deterministic, unit-testable citation builder for the four live-data
    tools - the same role should_chart plays for charts. Keyed on the
    context dict each tool records (_record_tool_call's third element),
    not the structured result, since what needs citing here is the query
    that was run, not the shape of what came back.

    Returns None for search_guide (cited separately, by chunk id/page - see
    ask()) and for a call with no context recorded (e.g. a failed lookup
    that returned early before _record_tool_call ran).
    """
    if not context:
        return None

    if tool_name == "lookup_agency":
        name = context["name"]
        return ToolCitation(
            tool_name=tool_name,
            parameters={"name": name},
            description=f"Agency lookup: {name}",
        )

    if tool_name == "get_spending_by_category":
        params = {
            "category": context["category"],
            "agency_name": context["agency_name"],
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
        }
        description = (
            f"{params['category']} breakdown, {params['agency_name']}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "get_spending_over_time":
        params = {
            "agency_name": context["agency_name"],
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
            "group": context["group"],
        }
        description = (
            f"Spending over time ({params['group']}), {params['agency_name']}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "search_awards":
        params = {
            "agency_name": context["agency_name"],
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
            "award_type": context["award_type"],
        }
        description = (
            f"{params['award_type']} awards search, {params['agency_name']}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "code_execution":
        # Cite the actual command that ran, not just "code was run" - a
        # user or auditor should be able to see what was computed and from
        # what, the same way a spending tool's citation shows the query
        # that was run, not just that a call succeeded.
        command = context.get("command", "")
        shown_command = command if len(command) <= 200 else f"{command[:200]}..."
        return ToolCitation(
            tool_name=tool_name,
            parameters={"command": command},
            description=f"Code execution: {shown_command}",
        )

    return None

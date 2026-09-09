"""Pure, unit-testable post-processing of already-returned tool results:
turning a structured tool result into a chart or a citation, and the
fiscal-year date math the live-data tools rely on. Deliberately has no
dependency on the Anthropic client or tool_runner - everything here operates
on plain data (structured results, context dicts) that's already been
returned from a tool call, not on how that call was made.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel

from backend.app.usaspending_client import USASpendingAPIError


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


EARLIEST_SUPPORTED_FISCAL_YEAR = 2008


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

    Raises USASpendingAPIError for a year outside [EARLIEST_SUPPORTED_FISCAL_YEAR, current+1].
    """
    ceiling = current_fiscal_year() + 1
    for label, year in (("start_fiscal_year", start_fiscal_year), ("end_fiscal_year", end_fiscal_year)):
        if not (EARLIEST_SUPPORTED_FISCAL_YEAR <= year <= ceiling):
            raise USASpendingAPIError(
                f"{label}={year} is out of range - USASpending data only covers "
                f"FY{EARLIEST_SUPPORTED_FISCAL_YEAR} through FY{ceiling}."
            )
    if start_fiscal_year > end_fiscal_year:
        raise USASpendingAPIError(
            f"start_fiscal_year={start_fiscal_year} is after end_fiscal_year={end_fiscal_year}."
        )

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
    # Exactly one of page/term/question is normally set. term: the
    # USASpending Glossary, paired with a real per-term `url` (see
    # GLOSSARY_URL_BASE below) since unlike the Guide, the live glossary
    # sidebar honors a per-term deep link. question: the majority of
    # Analyst's Guide chunks, which are Q&A pairs - carries the real
    # question text (extracted from the chunk itself), paired with `url`
    # linking to the live guide page. page: the minority of Guide chunks
    # that aren't Q&A-shaped (section headers, table of contents -
    # verified live only ~60% of Guide chunks match the Q&A pattern, not
    # assumed to be all of them) - also paired with `url`, since the live
    # Guide page is a fixed, non-anchor-addressable single URL either way
    # (verified live 2026-09-07: it's a client-rendered SPA with no
    # server-side per-question anchors at all, so `url` is the same for
    # every Guide citation regardless of which question - `question`/
    # `page` is what actually distinguishes them). Glossary citations
    # differ: the live glossary sidebar IS addressable per-term via a
    # query param, so `url` varies per citation there instead of being
    # fixed.
    page: int | None = None
    term: str | None = None
    question: str | None = None
    url: str | None = None


# The Analyst's Guide's live counterpart to the local PDF this project
# ingests. Fixed for every Guide citation - see Citation's docstring for
# why this can't be a per-question deep link.
GUIDE_URL = "https://www.usaspending.gov/federal-spending-guide"

# The live glossary is a sidebar popup on the main site, not a dedicated
# HTML page - but it opens pre-scrolled to a specific term via this query
# param, and this isn't a guess: it's the exact mechanism USASpending's own
# glossary entries use to cross-link each other internally (found live,
# 2026-09-07, in the real API response for the "treasury-account-symbol-tas"
# entry: its `resources` field contains a literal
# "[Federal Account](?glossary=federal-account)" link). Root path chosen
# over /federal-spending-guide as the base since the sidebar is a site-wide
# component, not Guide-specific - confirmed live that
# https://www.usaspending.gov/?glossary=treasury-account-symbol-tas opens
# the sidebar to that exact term.
GLOSSARY_URL_BASE = "https://www.usaspending.gov/?glossary="

# ingest.py's chunker splits the Guide on question boundaries. Most
# resulting Q&A chunks' text begins with the literal question wrapped in
# a matched pair of smart single quotes (e.g. "‘What is an obligation?’"),
# but not always - found live (2026-09-07, diffing our indexed chunks
# against the real usaspending.gov/federal-spending-guide page) two
# distinct ways a question can fail to have a clean quote pair, both
# confirmed directly against pymupdf's raw extracted text, not assumed:
# some questions have no leading quote at all (typically the first
# question right after a new section header, e.g. "What is a recipient?"
# under "RECIPIENT DATA ELEMENTS"), and some have an opening quote but no
# closing one (the source PDF mis-renders or drops the closing glyph -
# ingest.py's own QUESTION_START_RE comment already documents this same
# phenomenon for chunk *boundary* detection; it turns out to affect
# extracting the question text back out of an already-correctly-chunked
# unit too). Strips a leading and trailing quote independently, rather
# than requiring a matched pair, so both cases resolve to the same clean
# question text instead of only one of them working.
_GUIDE_QUESTION_PATTERN = re.compile(r"^[‘“]?(.+\?)[’”]?\s*$")


def _extract_guide_question(text: str) -> str | None:
    """Returns the real question a Guide chunk is answering, or None if
    this chunk isn't Q&A-shaped (a section header or table-of-contents
    chunk, not a definitional Q&A pair - such chunks don't end in "?" at
    all, so they never match). Returning None for those rather than
    fabricating a "question" from non-Q&A text is what lets the caller
    fall back to a page-number citation for exactly the chunks a question
    doesn't make sense for.
    """
    first_line = text.split("\n", 1)[0].strip()
    match = _GUIDE_QUESTION_PATTERN.match(first_line)
    return match.group(1).strip() if match else None


def _build_guide_citation(chunk: dict) -> Citation:
    """One retrieved chunk -> one Citation, encapsulating the term
    (Glossary) / question (Q&A-shaped Guide chunk) / page (non-Q&A Guide
    chunk) branching so it's unit-testable independent of ask()'s loop.
    """
    term = chunk.get("term")
    if term:
        slug = chunk.get("slug")
        url = f"{GLOSSARY_URL_BASE}{slug}" if slug else None
        return Citation(chunk_id=chunk["id"], source=chunk["source"], term=term, url=url)

    question = _extract_guide_question(chunk["text"])
    if question is not None:
        return Citation(chunk_id=chunk["id"], source=chunk["source"], question=question, url=GUIDE_URL)

    return Citation(chunk_id=chunk["id"], source=chunk["source"], page=chunk["page_start"], url=GUIDE_URL)


class ToolCitation(BaseModel):
    """Citation for a live USASpending.gov API call - the live-data analog
    of Citation. There's no chunk id or page number for a live query, so
    this doesn't try to force that shape; instead it cites the tool and the
    exact parameters used, which is what a user or auditor actually needs
    to re-run the same query and verify the numbers themselves."""

    tool_name: str
    parameters: dict[str, str | int | float]
    description: str


# Tools whose results are never chart-worthy by shape (free text / a single
# profile), regardless of what's in the result.
# get_agency_budget's multi-year results are chartable in principle (same
# cardinality shape as get_spending_over_time), but which of its three
# metrics (budgetary resources vs. obligated vs. outlayed) to chart is a
# real design question deferred for now - excluded here rather than
# guessing which one the model would want.
NEVER_CHART_TOOLS = {
    "search_guide", "lookup_agency", "search_awards", "get_agency_budget",
    "code_execution", "get_award_details",
    # search_recipients returns a ranked candidate list, plausibly
    # chart-worthy on its own merits (like get_spending_by_category) - but
    # its real purpose is disambiguation/resolution, not analysis, and
    # deciding whether that's ever worth charting is a separate, real
    # design question deferred rather than resolved as a side effect here
    # (2026-09-08). get_recipient_details is a single profile, no
    # cardinality to chart, same as get_award_details.
    "search_recipients", "get_recipient_details",
}


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

    if tool_name == "get_spending_by_geography":
        if len(structured_result.results) < 2:
            return None
        top = sorted(structured_result.results, key=lambda r: -r.aggregated_amount)[:20]
        title = f"Spending by {structured_result.geo_layer}"
        if agency_name:
            title += f" — {agency_name}"
        return ChartSpec(
            chart_type="bar",
            title=title,
            labels=[r.display_name or r.shape_code or "Unknown" for r in top],
            values=[r.aggregated_amount for r in top],
        )

    return None


# The six optional filter params get_spending_by_category, get_spending_over_time,
# and search_awards now all accept (see tools.py's _build_filters/
# _record_optional_filter_context) - present in a call's context dict only
# when actually set for that call, so a citation reflects exactly which
# filters were used, not every filter the tool supports in the abstract.
_ALL_OPTIONAL_FILTER_KEYS = {
    "agency_name",
    "award_type",
    "recipient_name",
    "recipient_id",
    "min_amount",
    "max_amount",
    "performed_in_state",
    "recipient_in_state",
}


def _citation_scope_label(context: dict) -> str:
    """What a citation's description names as what the query was scoped
    to. agency_name if given (existing behavior, preserved) - else
    recipient_name (human-readable) or recipient_id (a resolved but
    nameless identifier) if that's what scoped the call instead.
    _build_filters (tool_filters.py) guarantees at least one of the three
    is always present - "unknown scope" here would mean that guarantee
    was violated, not a real expected case."""
    return context.get("agency_name") or context.get("recipient_name") or context.get("recipient_id") or "unknown scope"


def _merge_optional_filter_params(params: dict, context: dict, keys: set[str]) -> None:
    for key in keys:
        if key in context:
            params[key] = context[key]


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

    if tool_name == "get_agency_budget":
        params = {
            "agency_name": context["agency_name"],
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
        }
        description = (
            f"Budgetary resources, {params['agency_name']}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "get_spending_by_category":
        params = {
            "category": context["category"],
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
        }
        _merge_optional_filter_params(params, context, _ALL_OPTIONAL_FILTER_KEYS)
        scope = _citation_scope_label(context)
        description = (
            f"{params['category']} breakdown, {scope}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "get_spending_over_time":
        params = {
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
            "group": context["group"],
        }
        _merge_optional_filter_params(params, context, _ALL_OPTIONAL_FILTER_KEYS)
        scope = _citation_scope_label(context)
        description = (
            f"Spending over time ({params['group']}), {scope}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "get_spending_by_geography":
        params = {
            "scope": context["scope"], "geo_layer": context["geo_layer"],
            "start_fiscal_year": context["start_fiscal_year"], "end_fiscal_year": context["end_fiscal_year"],
        }
        _merge_optional_filter_params(params, context, _ALL_OPTIONAL_FILTER_KEYS)
        scope = _citation_scope_label(context)
        description = (
            f"Spending by {params['geo_layer']} ({params['scope']}), {scope}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "search_awards":
        params = {
            "start_fiscal_year": context["start_fiscal_year"],
            "end_fiscal_year": context["end_fiscal_year"],
            "award_type": context["award_type"],
        }
        # award_type is already set above (unconditionally, unlike every
        # other key here including agency_name, which is now optional
        # too - see tool_filters.py's _build_filters) - merge everything
        # else.
        _merge_optional_filter_params(params, context, _ALL_OPTIONAL_FILTER_KEYS - {"award_type"})
        scope = _citation_scope_label(context)
        description = (
            f"{params['award_type']} awards search, {scope}, "
            f"FY{params['start_fiscal_year']}-FY{params['end_fiscal_year']}"
        )
        return ToolCitation(tool_name=tool_name, parameters=params, description=description)

    if tool_name == "get_award_details":
        award_id = context["award_id"]
        # piid falls back to the raw internal_id (context["award_id"])
        # only if the response somehow carried neither piid/fain/uri -
        # shouldn't happen per the live award_id.md contract, but keeps
        # the citation from showing "None" instead of something real.
        label = context.get("piid") or award_id
        return ToolCitation(
            tool_name=tool_name,
            parameters={"award_id": award_id},
            description=f"Award details: {label}",
        )

    if tool_name == "search_recipients":
        keyword = context["keyword"]
        return ToolCitation(
            tool_name=tool_name,
            parameters={"keyword": keyword},
            description=f"Recipient search: {keyword}",
        )

    if tool_name == "get_recipient_details":
        recipient_id = context["recipient_id"]
        # name falls back to the raw recipient_id only if the response
        # somehow carried no name at all - shouldn't happen per the live
        # recipient_id.md contract (name is nullable but real-world
        # recipients always have one, even the redacted/aggregate bucket
        # case, which has its own real sentinel string), but keeps the
        # citation from showing "None" instead of something real.
        label = context.get("name") or recipient_id
        return ToolCitation(
            tool_name=tool_name,
            parameters={"recipient_id": recipient_id},
            description=f"Recipient details: {label}",
        )

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

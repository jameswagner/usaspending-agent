"""The system prompt, the top-level result shape, and ask() - the
tool-calling loop plus the chart/citation extraction that runs over its
capture buffer afterward.
"""
from __future__ import annotations

from datetime import datetime, timezone

from langsmith import traceable
from pydantic import BaseModel

from .clients import MODEL, _get_client
from .response_shaping import (
    ChartSpec,
    Citation,
    ToolCitation,
    build_tool_citation,
    current_fiscal_year,
    should_chart,
)
from .scope import _is_in_scope
from .tools import (
    _tool_call_log,
    get_spending_by_category,
    get_spending_over_time,
    lookup_agency,
    search_awards,
    search_guide,
)

NOT_FOUND_MESSAGE = (
    "I can only answer questions about USASpending.gov federal spending data, "
    "and couldn't find anything relevant to this question."
)


def _build_system_prompt() -> str:
    """Rebuilt on every ask() call (not a module-level constant) so the date
    grounding below is never stale in a long-running server process - see
    current_fiscal_year's docstring for the bug this closes: the model has
    no other way to know "today," so relative phrases like "most recent
    fiscal year" were previously guessed from training data instead of
    computed, and answered FY2024 when FY2025 data was already live.
    """
    today = datetime.now(timezone.utc).date()
    current_fy = current_fiscal_year(today)
    most_recent_completed_fy = current_fy - 1

    return (
        "You answer questions about USASpending.gov federal spending data. You "
        "have five tools: search_guide (conceptual/definitional questions about "
        "USASpending data, terms, and fields), lookup_agency (what a specific "
        "federal agency is, or its toptier code), get_spending_by_category "
        "(an agency's spending broken down by NAICS/PSC/sub-agency/etc. for a "
        "fiscal year range), get_spending_over_time (an agency's spending trend "
        "across fiscal years/quarters/months), and search_awards (individual "
        "contract/grant/loan records for an agency and fiscal year range — use "
        "this for 'show me awards from X' or 'who received money from X', not "
        "for aggregate breakdowns or trends). You must call at least one of "
        "these tools before writing any answer, for every question, with no "
        "exceptions — including questions that seem unrelated to federal "
        "spending, general-knowledge questions, greetings, or anything else. "
        "Never answer from your own knowledge without calling a tool first, "
        "even if you already know the answer. Base your answer strictly on what "
        "the tools return. If no tool finds anything relevant, or the question "
        "has nothing to do with USASpending federal spending data, tell the "
        "user plainly that you can only answer questions about USASpending "
        "data — do not answer the question anyway. If a specific tool call "
        "fails or the exact breakdown/data requested isn't available, say so "
        "plainly. Do not silently substitute a different category, agency, or "
        "time period and present those results as if they answered the "
        "original question — if you use different parameters than what was "
        "asked because the exact request failed, say so explicitly. These "
        "tools use federal fiscal years (FY2021 = October 2020-September "
        "2021, named by the year it ends in) — always label years explicitly "
        "as fiscal years (e.g. 'FY2021' or 'fiscal year 2021') in your answer, "
        "never a bare year number, since it is easily confused with a "
        f"calendar year. Today's date is {today.isoformat()}, so the current "
        f"(in-progress, incomplete) fiscal year is FY{current_fy} and the most "
        f"recently *completed* fiscal year is FY{most_recent_completed_fy}. "
        "When a question says 'most recent,' 'latest,' 'current,' 'this "
        "year,' or similar with no fiscal year stated explicitly, use these "
        "values — do not guess a year from your own training data, since it "
        "does not know today's actual date. 'Most recent complete fiscal "
        "year' normally means the one that just ended, not the one in "
        "progress; only use the in-progress fiscal year if the question is "
        "explicitly about data so far this year."
    )


class AgentResult(BaseModel):
    answer_text: str
    charts: list[ChartSpec] = []
    citations: list[Citation] = []
    tool_citations: list[ToolCitation] = []


@traceable(run_type="chain", name="agent_ask")
def ask(question: str) -> AgentResult:
    if not _is_in_scope(question):
        return AgentResult(answer_text=NOT_FOUND_MESSAGE)

    _tool_call_log.set([])

    runner = _get_client().beta.messages.tool_runner(
        model=MODEL,
        max_tokens=2048,
        system=_build_system_prompt(),
        tools=[
            search_guide,
            lookup_agency,
            get_spending_by_category,
            get_spending_over_time,
            search_awards,
        ],
        messages=[{"role": "user", "content": question}],
    )

    final = None
    for message in runner:
        final = message

    answer_text = next((b.text for b in final.content if b.type == "text"), "")

    # One chart per chart-worthy tool call in the turn (e.g. a "compare NSF
    # and Education's spending trend" question makes two get_spending_over_time
    # calls, each its own chart) - previously took only the first and
    # silently dropped the rest, which a real two-agency comparison question
    # surfaced immediately. Deliberately not merged into one multi-series
    # chart: two calls aren't guaranteed to cover the same fiscal years, and
    # aligning them onto one shared axis is real logic this doesn't attempt.
    #
    # Guide citations (chunk id/source/page) and live-data citations (tool +
    # query parameters, since there's no "page" for a live lookup) are both
    # built from the same capture buffer. Verifying any arithmetic the model
    # does on top of retrieved numbers is a separate, harder problem,
    # deferred (see BACKLOG.md).
    charts: list[ChartSpec] = []
    seen_chunk_ids: set[str] = set()
    citations: list[Citation] = []
    seen_tool_citation_keys: set[tuple] = set()
    tool_citations: list[ToolCitation] = []
    for tool_name, result, context in _tool_call_log.get() or []:
        chart = should_chart(tool_name, result, context)
        if chart is not None:
            charts.append(chart)

        if tool_name == "search_guide":
            for chunk in result:
                if chunk["id"] in seen_chunk_ids:
                    continue
                seen_chunk_ids.add(chunk["id"])
                citations.append(Citation(chunk_id=chunk["id"], source=chunk["source"], page=chunk["page_start"]))
            continue

        tool_citation = build_tool_citation(tool_name, context)
        if tool_citation is None:
            continue
        dedup_key = (tool_citation.tool_name, tuple(sorted(tool_citation.parameters.items())))
        if dedup_key in seen_tool_citation_keys:
            continue
        seen_tool_citation_keys.add(dedup_key)
        tool_citations.append(tool_citation)

    return AgentResult(answer_text=answer_text, charts=charts, citations=citations, tool_citations=tool_citations)

"""Tool-calling agent over USASpending question-answering.

First tool: search_guide (wraps HybridRetriever), matching the current
/ask behavior. Live-data tools (agency lookup, spending queries, award
search) are added one at a time after this is verified.

Usage:
  python -m backend.app.agent --question "What is a prime award?"
"""
from __future__ import annotations

import os

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv
from langsmith import traceable

from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.tools.usaspending_client import (
    AdvancedFilters,
    AgencyFilter,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

load_dotenv()

ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID")

# Swappable via env var so Haiku 4.5 vs Sonnet 5 can be compared on the same
# test questions before picking one for tool-calling specifically.
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")

# Same floor established in main.py / sanity_check.py: below this, rerank
# scores mean "nothing relevant" rather than a weak match.
RERANK_CONFIDENCE_THRESHOLD = -5.0

_client: anthropic.Anthropic | None = None
_retriever: HybridRetriever | None = None
_usaspending_client: USASpendingClient | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
        _client = anthropic.Anthropic(default_headers=headers)
    return _client


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_usaspending_client() -> USASpendingClient:
    global _usaspending_client
    if _usaspending_client is None:
        _usaspending_client = USASpendingClient()
    return _usaspending_client


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
    return (
        f"Agency: {overview.name} ({overview.abbreviation})\n"
        f"Toptier code: {overview.toptier_code}\n"
        f"Fiscal year: {overview.fiscal_year}\n"
        f"Subtier agency count: {overview.subtier_agency_count}\n"
        f"Mission: {overview.mission or 'N/A'}\n"
        f"Website: {overview.website or 'N/A'}"
    )


@beta_tool
def get_spending_by_category(
    category: str,
    agency_name: str,
    start_date: str,
    end_date: str,
    limit: int = 5,
) -> str:
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for one awarding agency and date range, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient_duns, state_territory. (Only these are live-verified; other category names the API contract lists, like object_class or tas, 404 in practice.)
        agency_name: The awarding agency's name, e.g. "National Science Foundation".
        start_date: Start of the date range, YYYY-MM-DD. Data is only available from 2007-10-01 onward.
        end_date: End of the date range, YYYY-MM-DD.
        limit: Max number of results to return (default 5).
    """
    client = _get_usaspending_client()
    filters = AdvancedFilters(
        agencies=[AgencyFilter(type="awarding", tier="toptier", name=agency_name)],
        time_period=[TimePeriod(start_date=start_date, end_date=end_date)],
    )
    try:
        response = client.spending_by_category(category, filters, limit=limit)
    except USASpendingAPIError as e:
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    if not response.results:
        return f"No {category} spending data found for {agency_name} between {start_date} and {end_date}."

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    return "\n".join(lines)


NOT_FOUND_MESSAGE = (
    "I can only answer questions about USASpending.gov federal spending data, "
    "and couldn't find anything relevant to this question."
)


SCOPE_CLASSIFIER_PROMPT = (
    "You classify whether a user question is in scope for a USASpending.gov "
    "assistant: federal spending, budgets, obligations/outlays, contracts, "
    "grants/financial assistance, awards, recipients, federal agencies, or "
    "USASpending.gov data/fields/API concepts. Respond with only YES or NO, "
    "nothing else."
)


@traceable(run_type="llm", name="scope_classifier")
def _is_in_scope(question: str) -> bool:
    """Cheap pre-filter gate: only start the (much more expensive) tool-
    calling loop if the question is plausibly in-scope for this app's whole
    domain, instead of relying on the system prompt alone to stop the model
    from answering off-topic questions from its own knowledge.

    Deliberately broader than "the guide has relevant content" — a
    live-data question (e.g. an agency lookup) can be legitimately in scope
    without matching anything in the static guide.
    """
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=SCOPE_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")


SYSTEM_PROMPT = (
    "You answer questions about USASpending.gov federal spending data. You "
    "have three tools: search_guide (conceptual/definitional questions about "
    "USASpending data, terms, and fields), lookup_agency (what a specific "
    "federal agency is, or its toptier code), and get_spending_by_category "
    "(an agency's spending broken down by NAICS/PSC/sub-agency/etc. for a "
    "date range). You must call at least one of these tools before writing "
    "any answer, for every question, with no "
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
    "asked because the exact request failed, say so explicitly."
)


@traceable(run_type="chain", name="agent_ask")
def ask(question: str) -> str:
    if not _is_in_scope(question):
        return NOT_FOUND_MESSAGE

    runner = _get_client().beta.messages.tool_runner(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[search_guide, lookup_agency, get_spending_by_category],
        messages=[{"role": "user", "content": question}],
    )

    final = None
    for message in runner:
        final = message

    return next((b.text for b in final.content if b.type == "text"), "")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    print(f"[model={MODEL}]")
    print(ask(args.question))


if __name__ == "__main__":
    main()

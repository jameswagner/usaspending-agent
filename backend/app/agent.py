"""Tool-calling agent over USASpending question-answering.

First tool: search_guide (wraps HybridRetriever), matching the current
/ask behavior. Live-data tools (agency lookup, spending queries, award
search) are added one at a time after this is verified.

Usage:
  python -m backend.app.agent --question "What is a prime award?"
"""
from __future__ import annotations

import os
from typing import Optional

import anthropic
from anthropic import beta_tool
from dotenv import load_dotenv
from langsmith import traceable

from backend.app.retrieval.hybrid import HybridRetriever

load_dotenv()

ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID")

# Swappable via env var so Haiku 4.5 vs Sonnet 5 can be compared on the same
# test questions before picking one for tool-calling specifically.
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")

# Same floor established in main.py / sanity_check.py: below this, rerank
# scores mean "nothing relevant" rather than a weak match.
RERANK_CONFIDENCE_THRESHOLD = -5.0

_client: Optional[anthropic.Anthropic] = None
_retriever: Optional[HybridRetriever] = None


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


SYSTEM_PROMPT = (
    "You answer questions about USASpending.gov federal spending data. "
    "You must call the search_guide tool at least once before writing any "
    "answer, for every question, with no exceptions — including questions "
    "that seem unrelated to federal spending, general-knowledge questions, "
    "greetings, or anything else. Never answer from your own knowledge "
    "without calling the tool first, even if you already know the answer. "
    "Base your answer strictly on what the tool returns. If the tool finds "
    "nothing relevant, or the question has nothing to do with USASpending "
    "federal spending data, tell the user plainly that you can only answer "
    "questions about USASpending data — do not answer the question anyway."
)


@traceable(run_type="chain", name="agent_ask")
def ask(question: str) -> str:
    runner = _get_client().beta.messages.tool_runner(
        model=MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        tools=[search_guide],
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

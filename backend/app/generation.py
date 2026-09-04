"""LLM-based answer synthesis over retrieved Analyst's Guide chunks.

Takes the chunks HybridRetriever already found and has Claude write a real
answer grounded in them, instead of returning raw chunk text. No tool use,
no agent loop — a single Claude call per question.
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID")

# Cheapest available model, per explicit request — not the skill-default
# Opus 5. Revisit if answer quality on harder guide questions is poor.
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You answer questions about USASpending.gov federal spending data using "
    "only the excerpts provided from the Analyst's Guide to Federal Spending "
    "Data. Write a direct, concise answer grounded strictly in the excerpts. "
    "If the excerpts don't actually answer the question, say so plainly "
    "instead of guessing or using outside knowledge."
)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        # Some API keys are "identity-linked" and require the target
        # workspace to be sent explicitly; the SDK has no dedicated
        # constructor param for this, so it goes in as a raw header.
        headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
        _client = anthropic.Anthropic(default_headers=headers)
    return _client


def _format_context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(f"[Page {c['page_start']}]\n{c['text']}" for c in chunks)


@traceable(run_type="llm", name="synthesize_answer")
def synthesize_answer(question: str, chunks: list[dict]) -> str:
    context = _format_context(chunks)
    response = _get_client().messages.create(
        model=MODEL,
        # Deliberately short output: synthesized guide answers run a few
        # sentences to a short paragraph, not long-form content.
        max_tokens=2048,
        # No output_config/effort here — Haiku 4.5 doesn't support it
        # (that's an Opus/Sonnet-5-tier feature).
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Excerpts from the Analyst's Guide:\n\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return next((block.text for block in response.content if block.type == "text"), "")

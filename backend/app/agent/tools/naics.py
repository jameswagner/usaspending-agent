"""resolve_naics_code - semantic lookup from a plain-English industry
description to a NAICS code, for search_awards/get_spending_by_category's
naics_code filter. See #6: not search_guide (definitional Q&A) - this
resolves to a code a follow-up tool call then uses, same two-step shape as
search_recipients -> get_recipient_details.
"""
from __future__ import annotations

from anthropic import beta_tool

from backend.app.usaspending import USASpendingAPIError

from ..singletons import (
    RERANK_CONFIDENCE_THRESHOLD,
    _get_naics_retriever,
    _get_usaspending_client,
)
from ._shared import _record_tool_call, _wrap_untrusted


def _list_naics_children(code: str) -> str:
    try:
        children = _get_usaspending_client().get_naics_children(code)
    except USASpendingAPIError as e:
        return f"This query failed: {e}."
    _record_tool_call(
        "resolve_naics_code", [{"slug": c.naics, "term": c.naics_description} for c in children], {"description": code}
    )
    if not children:
        return f"NAICS {code} has no sub-industries listed."
    lines = [f"{c.naics} - {c.naics_description}" + (f" (retired {c.year_retired})" if c.year_retired else "") for c in children]
    return _wrap_untrusted(f"Sub-industries directly under NAICS {code} (official list, complete):\n" + "\n".join(lines))


@beta_tool
def resolve_naics_code(description: str) -> str:
    """Find NAICS code(s) matching a plain-English industry/business description, e.g. "custom software development" or "ship building". Use this before naics_code on search_awards/get_spending_by_category/get_spending_over_time whenever the question describes an industry rather than already naming a specific code. To list the industries under a code ("what falls under NAICS 54"), pass the 2- or 4-digit code itself as the description (sectors list 4-digit groups, which list 6-digit industries).

    Results are semantic matches against the official NAICS description/index text, not exact/authoritative lookups - always confirm the returned code's title actually matches what the question meant before using it as a filter, and say so if presenting a match to the user (e.g. "closest matching NAICS code, not a confirmed exact match"). A code passed in returns its official sub-industries instead.

    Args:
        description: A plain-English industry or business description, or a 2- or 4-digit NAICS code to list the industries under it.
    """
    code = description.strip()
    if code.isdigit() and len(code) in (2, 4):
        return _list_naics_children(code)
    results = _get_naics_retriever().retrieve(description, top_k=5)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]
    if not matches:
        return "No NAICS code found matching this description."

    _record_tool_call("resolve_naics_code", matches, {"description": description})

    lines = [f"{m['slug']} - {m['term']}" for m in matches]
    return _wrap_untrusted(
        "Best semantic matches, not confirmed exact matches - verify the title fits before using a code:\n"
        + "\n".join(lines)
    )

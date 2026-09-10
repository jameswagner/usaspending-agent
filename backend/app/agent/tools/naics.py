"""resolve_naics_code - semantic lookup from a plain-English industry
description to a NAICS code, for search_awards/get_spending_by_category's
naics_code filter. See #6: not search_guide (definitional Q&A) - this
resolves to a code a follow-up tool call then uses, same two-step shape as
search_recipients -> get_recipient_details.
"""
from __future__ import annotations

from anthropic import beta_tool

from ..singletons import RERANK_CONFIDENCE_THRESHOLD, _get_naics_retriever
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted


@beta_tool
def resolve_naics_code(description: str) -> str:
    """Find NAICS code(s) matching a plain-English industry/business description, e.g. "custom software development" or "ship building". Use this before naics_code on search_awards/get_spending_by_category/get_spending_over_time whenever the question describes an industry rather than already naming a specific code.

    Results are semantic matches against the official NAICS description/index text, not exact/authoritative lookups - always confirm the returned code's title actually matches what the question meant before using it as a filter, and say so if presenting a match to the user (e.g. "closest matching NAICS code, not a confirmed exact match").

    Args:
        description: A plain-English industry or business description.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
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

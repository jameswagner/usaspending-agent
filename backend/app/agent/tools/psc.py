"""resolve_psc_code - semantic lookup from a plain-English product/service
description to a PSC code, for search_awards/get_spending_by_category's
psc_code filter. Same shape as naics.py's resolve_naics_code - see #6.
"""
from __future__ import annotations

from anthropic import beta_tool

from ..singletons import RERANK_CONFIDENCE_THRESHOLD, _get_psc_retriever
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted


@beta_tool
def resolve_psc_code(description: str) -> str:
    """Find PSC (Product and Service Code) code(s) matching a plain-English product/service description, e.g. "aircraft parts" or "systems engineering services". Use this before psc_code on search_awards/get_spending_by_category/get_spending_over_time whenever the question describes a product or service rather than already naming a specific code.

    Results are semantic matches against the official PSC description/includes/excludes text, not exact/authoritative lookups - always confirm the returned code's title actually matches what the question meant before using it as a filter, and say so if presenting a match to the user (e.g. "closest matching PSC code, not a confirmed exact match").

    Args:
        description: A plain-English product or service description.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    results = _get_psc_retriever().retrieve(description, top_k=5)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]
    if not matches:
        return "No PSC code found matching this description."

    _record_tool_call("resolve_psc_code", matches, {"description": description})

    lines = [f"{m['slug']} - {m['term']}" for m in matches]
    return _wrap_untrusted(
        "Best semantic matches, not confirmed exact matches - verify the title fits before using a code:\n"
        + "\n".join(lines)
    )

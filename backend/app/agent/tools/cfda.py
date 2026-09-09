"""resolve_cfda_program - semantic lookup from a plain-English description to
a CFDA/Assistance Listing program number, for search_awards/
get_spending_by_category's cfda_program filter. Same shape as naics.py's
resolve_naics_code and psc.py's resolve_psc_code - see #6.
"""
from __future__ import annotations

from anthropic import beta_tool

from ..singletons import RERANK_CONFIDENCE_THRESHOLD, _get_cfda_retriever
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted


@beta_tool
def resolve_cfda_program(description: str) -> str:
    """Find CFDA/Assistance Listing program number(s) matching a plain-English description of a federal grant, loan, or assistance program, e.g. "Medicaid" or "rural broadband grants". Use this before cfda_program on search_awards/get_spending_by_category/get_spending_over_time whenever the question describes a program rather than already naming a specific program number.

    Results are semantic matches against the official program title/objectives/uses text, not exact/authoritative lookups - always confirm the returned program's title actually matches what the question meant before using it as a filter, and say so if presenting a match to the user (e.g. "closest matching CFDA program, not a confirmed exact match").

    Args:
        description: A plain-English description of a federal assistance program.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    results = _get_cfda_retriever().retrieve(description, top_k=5)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]
    if not matches:
        return "No CFDA program found matching this description."

    _record_tool_call("resolve_cfda_program", matches)

    lines = [f"{m['slug']} - {m['term']}" for m in matches]
    return _wrap_untrusted(
        "Best semantic matches, not confirmed exact matches - verify the title fits before using a program number:\n"
        + "\n".join(lines)
    )

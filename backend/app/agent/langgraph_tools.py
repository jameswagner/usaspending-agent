"""LangChain-compatible equivalents of the @beta_tool-decorated functions,
for create_react_agent (#61). @beta_tool's BetaFunctionTool exposes the
original callable via .func - re-wrapping that with langchain_core's
tool(parse_docstring=True) reuses the exact same docstring/type hints,
verified against all 24 tools in use (including typing.Literal params,
which both beta_tool and LangChain translate into a JSON-schema enum via
Pydantic). The original @beta_tool-wrapped functions in tools/*.py are
untouched - still needed for tests, dev_tools scripts, and the legacy
ask() path during the migration window (see issue #61's series).

code_execution is deliberately not included - it's an Anthropic
server-side tool (a dict spec), not a Python callable, so there's
nothing to unwrap. Whether ChatAnthropic's tool-binding passes such a
dict through unmodified is a separate spike (issue #65).
"""
from __future__ import annotations

from langchain_core.tools import tool as _lc_tool

from .arithmetic_tools import (
    average,
    delta,
    percentage_of,
    rank_values,
    ratio,
    sum_values,
)
from .tools import (
    get_agency_award_breakdown,
    get_agency_budget,
    get_award_details,
    get_award_subawards,
    get_recipient_details,
    get_spending_by_category,
    get_spending_by_geography,
    get_spending_over_time,
    list_top_agencies_by_budget,
    lookup_agency,
    resolve_cfda_program,
    resolve_county_fips,
    resolve_naics_code,
    resolve_psc_code,
    search_awards,
    search_guide,
    search_recipients,
    search_subawards,
)

_BETA_TOOLS = [
    get_agency_award_breakdown,
    get_agency_budget,
    get_award_details,
    get_award_subawards,
    get_recipient_details,
    get_spending_by_category,
    get_spending_by_geography,
    get_spending_over_time,
    list_top_agencies_by_budget,
    lookup_agency,
    resolve_cfda_program,
    resolve_county_fips,
    resolve_naics_code,
    resolve_psc_code,
    search_awards,
    search_guide,
    search_recipients,
    search_subawards,
]
_ARITHMETIC_TOOLS = [average, delta, percentage_of, rank_values, ratio, sum_values]

LANGGRAPH_TOOLS = [_lc_tool(bt.func, parse_docstring=True) for bt in _BETA_TOOLS + _ARITHMETIC_TOOLS]

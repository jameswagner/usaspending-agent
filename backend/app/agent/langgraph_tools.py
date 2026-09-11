"""LangChain-compatible equivalents of the @beta_tool-decorated functions,
for create_react_agent. @beta_tool's BetaFunctionTool exposes the
original callable via .func - re-wrapping that with
tool(parse_docstring=True) reuses the same docstring/type hints.

code_execution is deliberately not included: ChatAnthropic.bind_tools()
passes its dict spec through fine, but the resulting result blocks land
in AIMessage.content as plain dicts, not the attribute-access objects
_record_code_execution_calls (tools/_shared.py) expects, so citations for
it wouldn't work without adapting that function first.
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
    list_top_agencies_by_spending,
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
    list_top_agencies_by_spending,
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

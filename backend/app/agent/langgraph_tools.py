"""LangChain-compatible equivalents of the @beta_tool-decorated functions,
for create_react_agent. @beta_tool's BetaFunctionTool exposes the
original callable via .func - re-wrapping that with
tool(parse_docstring=True) reuses the same docstring/type hints.

code_execution is deliberately not included: ChatAnthropic.bind_tools()
passes its dict spec through fine, but the resulting result blocks land
in AIMessage.content as plain dicts, not the attribute-access objects
_record_code_execution_calls (tools/_shared.py) expects, so citations for
it wouldn't work without adapting that function first.

search_awards/search_subawards/search_transactions/get_spending_by_category/
get_spending_over_time/get_spending_by_geography are deliberately not bound
here either - query_spending consolidates all six into one tool; the six
stay real, callable, @beta_tool-decorated functions (query_spending
delegates to them) but aren't in _BETA_TOOLS, so the model never sees
their own schemas.

get_agency_budget_by_subcomponent is likewise not bound - get_agency_budget's
own group_by="sub_component" branch delegates to it, same reasoning as
query_spending above, but it isn't in _BETA_TOOLS itself.
"""
from __future__ import annotations

from langchain_core.tools import tool as _lc_tool
from pydantic import ConfigDict

from .arithmetic_tools import (
    average,
    delta,
    percentage_of,
    rank_values,
    ratio,
    sum_values,
)
from .tools import (
    download_records,
    download_single_award,
    get_agency_award_breakdown,
    get_agency_budget,
    get_award_details,
    get_award_funding_breakdown,
    get_award_subawards,
    get_award_transaction_history,
    get_award_type_breakdown,
    get_disaster_spending_overview,
    get_recipient_children,
    get_recipient_details,
    get_spending_explorer_breakdown,
    list_top_agencies_by_budget,
    lookup_agency,
    query_spending,
    resolve_budget_function,
    resolve_cfda_program,
    resolve_county_fips,
    resolve_naics_code,
    resolve_psc_code,
    search_guide,
    search_recipients,
)

_BETA_TOOLS = [
    download_records,
    download_single_award,
    get_agency_award_breakdown,
    get_agency_budget,
    get_award_details,
    get_award_funding_breakdown,
    get_award_subawards,
    get_award_transaction_history,
    get_award_type_breakdown,
    get_disaster_spending_overview,
    get_recipient_children,
    get_recipient_details,
    get_spending_explorer_breakdown,
    list_top_agencies_by_budget,
    lookup_agency,
    query_spending,
    resolve_budget_function,
    resolve_cfda_program,
    resolve_county_fips,
    resolve_naics_code,
    resolve_psc_code,
    search_guide,
    search_recipients,
]
_ARITHMETIC_TOOLS = [average, delta, percentage_of, rank_values, ratio, sum_values]

LANGGRAPH_TOOLS = [
    _lc_tool(bt.func, parse_docstring=bt not in (download_records, download_single_award))
    for bt in _BETA_TOOLS + _ARITHMETIC_TOOLS
]


def _reject_invalid_args(error, prefix: str) -> str:
    fields = sorted({
        ".".join(str(part) for part in item["loc"])
        for item in error.errors()
        if item["type"] == "extra_forbidden"
    })
    if fields:
        return f"{prefix} unsupported tool argument(s): {', '.join(fields)}. Check the tool schema."
    return f"{prefix} invalid tool arguments. Check the tool schema."


for _wrapped in LANGGRAPH_TOOLS:
    if _wrapped.name in {"query_spending", "download_records", "download_single_award"}:
        _wrapped.args_schema = type(
            f"Strict{_wrapped.name}Args",
            (_wrapped.args_schema,),
            {"model_config": ConfigDict(arbitrary_types_allowed=True, extra="forbid")},
        )
        _prefix = "This query failed:" if _wrapped.name == "query_spending" else "This download failed:"
        _wrapped.handle_validation_error = lambda error, prefix=_prefix: _reject_invalid_args(error, prefix)

"""The @beta_tool-decorated functions the agent calls, and their _raw
variants (structured Pydantic responses, presentation-free) that also feed
should_chart/build_tool_citation after the tool-calling loop finishes.

The _raw variants (not the @beta_tool wrappers) are @traceable. Tried
stacking @traceable directly under @beta_tool first and it broke tool
schemas: traceable injects its own optional `config` kwarg into the
wrapped function's signature, and beta_tool's schema generation picked it
up as a real, model-visible tool parameter. The _raw functions were never
@beta_tool in the first place, so tracing them is risk-free and also puts
the trace where the actual work (agency resolution, API calls) happens,
not on the thin formatting wrapper around it.
"""
from __future__ import annotations

from ._shared import (
    MAX_TOOL_CALLS_PER_TURN,
    _check_tool_call_budget,
    _format_agency_award_breakdown,
    _format_api_messages,
    _format_period_breakdown,
    _format_top_agencies_by_budget,
    _record_code_execution_calls,
    _record_tool_call,
    _scope_label,
    _tool_call_log,
    _truncation_note,
    _wrap_untrusted,
    get_agency_award_breakdown,
    get_agency_award_breakdown_raw,
    get_agency_budget,
    get_agency_budget_raw,
    list_top_agencies_by_budget,
    logger,
    lookup_agency,
    search_guide,
)
from .awards import (
    _agency_label,
    _format_award_details,
    _format_contract_or_idv,
    _format_financial_assistance,
    _format_period_of_performance,
    _location_label,
    get_award_details,
    get_award_details_raw,
    get_idv_amounts_raw,
)
from .business_type_labels import _format_business_type
from .cfda import resolve_cfda_program
from .naics import resolve_naics_code
from .psc import resolve_psc_code
from .recipients import (
    _format_recipient_address,
    _format_recipient_level,
    _format_recipient_listing,
    _format_recipient_overview,
    _format_recipient_state_only,
    get_recipient_details,
    search_recipients,
)
from .spending import (
    VALID_CATEGORIES,
    VALID_GROUPS,
    Category,
    Group,
    _format_geography_result,
    _normalize_category,
    _normalize_group,
    get_spending_by_category,
    get_spending_by_category_raw,
    get_spending_by_geography,
    get_spending_by_geography_raw,
    get_spending_over_time,
    get_spending_over_time_raw,
    search_awards,
    search_awards_raw,
)

__all__ = [
    "MAX_TOOL_CALLS_PER_TURN",
    "VALID_CATEGORIES",
    "VALID_GROUPS",
    "Category",
    "Group",
    "_agency_label",
    "_check_tool_call_budget",
    "_format_agency_award_breakdown",
    "_format_api_messages",
    "_format_award_details",
    "_format_business_type",
    "_format_contract_or_idv",
    "_format_financial_assistance",
    "_format_geography_result",
    "_format_period_breakdown",
    "_format_period_of_performance",
    "_format_recipient_address",
    "_format_recipient_level",
    "_format_recipient_listing",
    "_format_recipient_overview",
    "_format_recipient_state_only",
    "_format_top_agencies_by_budget",
    "_location_label",
    "_normalize_category",
    "_normalize_group",
    "_record_code_execution_calls",
    "_record_tool_call",
    "_scope_label",
    "_tool_call_log",
    "_truncation_note",
    "_wrap_untrusted",
    "get_agency_award_breakdown",
    "get_agency_award_breakdown_raw",
    "get_agency_budget",
    "get_agency_budget_raw",
    "get_award_details",
    "get_award_details_raw",
    "get_idv_amounts_raw",
    "get_recipient_details",
    "get_spending_by_category",
    "get_spending_by_category_raw",
    "get_spending_by_geography",
    "get_spending_by_geography_raw",
    "get_spending_over_time",
    "get_spending_over_time_raw",
    "list_top_agencies_by_budget",
    "logger",
    "lookup_agency",
    "resolve_cfda_program",
    "resolve_naics_code",
    "resolve_psc_code",
    "search_awards",
    "search_awards_raw",
    "search_guide",
    "search_recipients",
]

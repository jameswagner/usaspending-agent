"""Pins build_tool_citation's per-tool output so its dispatch can be refactored
without changing a single citation."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.app.agent.response_shaping import build_tool_citation

GOLDEN_PATH = Path(__file__).parent / "golden" / "tool_citations.json"

_REQ = [("POST", "https://api.usaspending.gov/api/v2/search/spending_by_award/", {"filters": {"a": 1}})]
_FILTERS = {
    "agency_name": "National Science Foundation",
    "recipient_name": "MIT",
    "performed_in_state": "CA",
    "naics_code": "541511",
    "def_codes": ["L", "M"],
    "naics_auto_resolved": "auto-resolved from 'software'",
    "_requests": _REQ,
}
_RANGE = {"start_year": 2021, "end_year": 2024}
_NO_REQ_FILTERS = {k: v for k, v in _FILTERS.items() if k != "_requests"}

CASES: dict[str, tuple[str, dict, object]] = {
    "lookup_agency_with_result": ("lookup_agency", {"name": "NSF"}, SimpleNamespace(toptier_code="049")),
    "lookup_agency_no_result": ("lookup_agency", {"name": "NSF"}, None),
    "resolve_naics_with_result": ("resolve_naics_code", {"description": "software"}, [{"slug": "541511"}]),
    "resolve_naics_no_result": ("resolve_naics_code", {"description": "software"}, None),
    "resolve_psc_with_result": ("resolve_psc_code", {"description": "chairs"}, [{"slug": "7110"}]),
    "resolve_cfda": ("resolve_cfda_program", {"description": "research"}, [{"slug": "47.049"}]),
    "resolve_budget_function": ("resolve_budget_function", {"description": "defense"}, None),
    "resolve_county_fips": ("resolve_county_fips", {"description": "Cook County", "_requests": [("GET", "https://x/y", None)]}, None),
    "resolve_county_fips_no_requests": ("resolve_county_fips", {"description": "Cook County"}, None),
    "get_agency_budget": (
        "get_agency_budget",
        {"agency_name": "NSF", "start_fiscal_year": 2021, "end_fiscal_year": 2024, "toptier_code": "4900"},
        None,
    ),
    "agency_award_breakdown_default": (
        "get_agency_award_breakdown", {"agency_name": "NSF", "fiscal_year": 2024}, None,
    ),
    "agency_award_breakdown_category": (
        "get_agency_award_breakdown",
        {"agency_name": "NSF", "fiscal_year": 2024, "group_by": "award_category", "award_type": "grants"},
        None,
    ),
    "agency_award_breakdown_whole": (
        "get_agency_award_breakdown", {"agency_name": "NSF", "fiscal_year": 2024, "group_by": "whole_agency"}, None,
    ),
    "agency_award_breakdown_office": (
        "get_agency_award_breakdown",
        {"agency_name": "NSF", "fiscal_year": 2024, "group_by": "sub_agency", "sub_agency": "Office X"},
        None,
    ),
    "subcomponent_plain": ("get_agency_budget_by_subcomponent", {"agency_name": "NSF", "fiscal_year": 2024}, None),
    "subcomponent_bureau": (
        "get_agency_budget_by_subcomponent", {"agency_name": "NSF", "fiscal_year": 2024, "bureau": "B1"}, None,
    ),
    "disaster_no_defc": ("get_disaster_spending_overview", {"def_codes": []}, None),
    "disaster_with_defc": ("get_disaster_spending_overview", {"def_codes": ["L", "M"]}, None),
    "disaster_missing_defc_key": ("get_disaster_spending_overview", {"x": 1}, None),
    "award_type_breakdown": ("get_award_type_breakdown", {**_RANGE, **_FILTERS}, None),
    "award_type_breakdown_calendar": (
        "get_award_type_breakdown", {**_RANGE, "agency_name": "NSF", "time_period_type": "calendar"}, None,
    ),
    "category": ("get_spending_by_category", {**_RANGE, "category": "naics", **_FILTERS}, None),
    "category_award_level": (
        "get_spending_by_category",
        {**_RANGE, "category": "psc", "spending_level": "awards", "agency_name": "NSF"},
        None,
    ),
    "category_no_scope": ("get_spending_by_category", {**_RANGE, "category": "naics"}, None),
    "over_time": ("get_spending_over_time", {**_RANGE, "group": "fiscal_year", **_FILTERS}, None),
    "over_time_subawards": (
        "get_spending_over_time",
        {**_RANGE, "group": "month", "spending_level": "subawards", "recipient_name": "MIT"},
        None,
    ),
    "geography": (
        "get_spending_by_geography",
        {**_RANGE, "scope": "place_of_performance", "geo_layer": "state", "sort_by": "amount", **_FILTERS},
        None,
    ),
    "geography_no_sort": (
        "get_spending_by_geography",
        {**_RANGE, "scope": "recipient_location", "geo_layer": "county", "agency_name": "NSF"},
        None,
    ),
    "search_awards": ("search_awards", {**_RANGE, "award_type": "grants", **_FILTERS}, None),
    "search_subawards": ("search_subawards", {**_RANGE, "award_type": "contracts", **_FILTERS}, None),
    "search_transactions": ("search_transactions", {**_RANGE, "award_type": "all", **_FILTERS}, None),
    "search_transactions_fy_default": (
        "search_transactions", {**_RANGE, "award_type": "grants", "keywords": ["ai", "ml"]}, None,
    ),
    "award_details": ("get_award_details", {"award_id": "CONT_AWD_1", "piid": "P1"}, None),
    "award_details_no_piid": ("get_award_details", {"award_id": "CONT_AWD_1"}, None),
    "award_subawards": ("get_award_subawards", {"award_id": "CONT_AWD_1", "_requests": _REQ}, None),
    "award_funding": ("get_award_funding_breakdown", {"award_id": "CONT_AWD_1", "_requests": _REQ}, None),
    "award_transactions": ("get_award_transaction_history", {"award_id": "CONT_AWD_1", "_requests": _REQ}, None),
    "top_agencies": ("list_top_agencies_by_budget", {"limit": 5}, None),
    "explorer_minimal": (
        "get_spending_explorer_breakdown",
        {"group_by": "agency", "fiscal_year": 2024, "quarter": "3", "_requests": _REQ},
        None,
    ),
    "explorer_scoped": (
        "get_spending_explorer_breakdown",
        {"group_by": "object_class", "fiscal_year": 2024, "quarter": "4", "agency": "049", "budget_function": "270"},
        None,
    ),
    "search_recipients": ("search_recipients", {"keyword": "MIT", "_requests": _REQ}, None),
    "recipient_details": ("get_recipient_details", {"recipient_id": "abc-R", "name": "MIT", "year": 2024}, None),
    "recipient_details_defaults": ("get_recipient_details", {"recipient_id": "abc-R"}, None),
    "recipient_children": ("get_recipient_children", {"recipient_id": "abc-R", "name": "MIT"}, None),
    "recipient_children_no_name": ("get_recipient_children", {"recipient_id": "abc-R", "year": 2023}, None),
    "code_execution_short": ("code_execution", {"command": "print(1+1)"}, None),
    "code_execution_long": ("code_execution", {"command": "x" * 300}, None),
    "download_search": (
        "download_records",
        {
            "spending_level": "awards", "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "filters": SimpleNamespace(model_dump=lambda exclude_none: {"f": 1}),
            "columns": ["a", "b"], "api_spending_level": "awards", "flag": True, "list_field": ["x", "y"],
        },
        None,
    ),
    "download_search_with_request": (
        "download_records",
        {
            "spending_level": "transactions", "start_year": 2024, "end_year": 2024,
            "filters": None, "columns": [], "api_spending_level": "transactions",
            "_requests": [
                ("GET", "https://api.usaspending.gov/api/v2/other/", None),
                ("POST", "https://api.usaspending.gov/api/v2/download/awards/", {"k": 1}),
            ],
        },
        None,
    ),
    "download_single_award": (
        "download_single_award",
        {"award_id": "CONT_AWD_1", "endpoint": "contract", "spending_level": "awards"},
        None,
    ),
    "search_guide_returns_none": ("search_guide", {"query": "x"}, None),
    "unknown_tool": ("not_a_tool", {"x": 1}, None),
    "empty_context": ("lookup_agency", {}, None),
}


def _dump(citation) -> dict | None:
    return None if citation is None else citation.model_dump()


def compute_all() -> dict:
    return {name: _dump(build_tool_citation(tool, ctx, result)) for name, (tool, ctx, result) in CASES.items()}


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN_PATH.read_text())


def test_golden_covers_every_case(golden):
    assert set(golden) == set(CASES)


@pytest.mark.parametrize("case", sorted(CASES))
def test_citation_matches_golden(case, golden):
    tool, ctx, result = CASES[case]
    assert _dump(build_tool_citation(tool, ctx, result)) == golden[case]


def test_every_bound_tool_has_a_citation_builder_or_is_exempt():
    from backend.app.agent import response_shaping
    from backend.app.agent.langgraph_tools import _ARITHMETIC_TOOLS, LANGGRAPH_TOOLS

    # query_spending records its call under the underlying tool it delegates to; search_guide
    # is cited by chunk id/page in ask(); arithmetic tools make no API call.
    exempt = {"query_spending", "search_guide"} | {t.name for t in _ARITHMETIC_TOOLS}
    delegates = {
        "search_awards", "search_subawards", "search_transactions", "get_spending_by_category",
        "get_spending_over_time", "get_spending_by_geography", "get_agency_budget_by_subcomponent",
    }
    needed = ({t.name for t in LANGGRAPH_TOOLS} | delegates | {"code_execution"}) - exempt
    assert needed - set(response_shaping._CITATION_BUILDERS) == set()
    assert exempt & set(response_shaping._CITATION_BUILDERS) == set()

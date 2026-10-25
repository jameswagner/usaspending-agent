from __future__ import annotations

import pytest

from backend.app.agent.dev_tools.check_website_coverage import (
    LOCAL,
    PLUMBING,
    _matches,
    _segments,
    backend_endpoints,
    classify,
    load_tasks,
)


@pytest.fixture(scope="module")
def tasks():
    return {t["id"]: t for t in load_tasks()}


@pytest.fixture(scope="module")
def backend():
    return backend_endpoints()


@pytest.mark.parametrize(
    ("task_id", "status"),
    [
        ("Q001", "covered"),
        ("Q032", "covered"),
        ("Q121", "gap"),
        ("Q070", "gap"),
        ("Q104", "gap"),
        ("Q045", "plumbing"),
        ("Q052", "local"),
        ("Q085", "dead"),
        ("Q152", "ai_search"),
    ],
)
def test_classify(tasks, backend, task_id, status):
    assert classify(tasks[task_id], backend)[0] == status


def test_category_template_expands_to_each_category(backend):
    assert ["api", "v2", "search", "spending_by_category", "funding_subagency"] in backend
    assert not any("{category}" in b for b in backend)


def test_docstring_mentions_are_not_endpoints(backend):
    assert ["api", "v2", "autocomplete", "naics"] not in backend


def test_placeholder_only_matches_a_backend_placeholder():
    assert _matches(["awards", "{{contract_id}}"], ["awards", "{award_id}"], lambda s: s.startswith("{"))
    assert not _matches(["awards", "{{contract_id}}"], ["awards", "funding"], lambda s: s.startswith("{"))


def test_backend_placeholder_does_not_swallow_an_endpoint_name():
    def placeholder(s):
        return s.startswith("{")

    assert not _matches(["awards", "funding_rollup"], ["awards", "{award_id}"], placeholder, identifiers_only=True)
    assert _matches(["agency", "080"], ["agency", "{toptier_code}"], placeholder, identifiers_only=True)


@pytest.mark.parametrize("pattern", [*PLUMBING, *LOCAL])
def test_every_listed_pattern_matches_an_inventory_task(tasks, pattern):
    assert any(_matches(_segments(t["request"]["path"]), _segments(pattern), lambda s: s == "*") for t in tasks.values())


# The website's Advanced Search filters (usaspending-website src/js/dataMapping/search/awardsOperationKeys.js rootKeys)
# and the tool parameter that sets each one.
WEBSITE_SEARCH_FILTERS = {
    "keywords": "keywords",
    "description": "description",
    "time_period": "start_year",
    "award_type_codes": "award_type",
    "agencies": "agency_name",
    "treasury_account_components": "federal_account",
    "tas_codes": "tas_code",
    "recipient_search_text": "recipient_name",
    "recipient_scope": "recipient_scope",
    "recipient_locations": "recipient_in_state",
    "recipient_type_names": "recipient_type",
    "place_of_performance_scope": "place_of_performance_scope",
    "place_of_performance_locations": "performed_in_state",
    "award_amounts": "min_amount",
    "award_ids": "award_id",
    "program_numbers": "cfda_program",
    "naics_codes": "naics_code",
    "psc_codes": "psc_code",
    "def_codes": "def_codes",
    "contract_pricing_type_codes": "contract_pricing_type",
    "set_aside_type_codes": "set_aside_type",
    "extent_competed_type_codes": "extent_competed_type",
}


@pytest.fixture(scope="module")
def query_spending_args():
    from backend.app.agent.consolidated_tools import CONSOLIDATED_TOOLS
    from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS

    return {name: set(next(t for t in tools if t.name == "query_spending").args)
            for name, tools in (("current", LANGGRAPH_TOOLS), ("consolidated", CONSOLIDATED_TOOLS))}


@pytest.mark.parametrize(("field", "param"), WEBSITE_SEARCH_FILTERS.items())
def test_query_spending_exposes_every_website_search_filter(query_spending_args, field, param):
    for toolset, args in query_spending_args.items():
        assert param in args, f"{toolset} query_spending has no {param} for the website's {field} filter"

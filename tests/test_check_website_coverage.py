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


@pytest.mark.parametrize("pattern", [*PLUMBING, *LOCAL])
def test_every_listed_pattern_matches_an_inventory_task(tasks, pattern):
    assert any(_matches(_segments(t["request"]["path"]), _segments(pattern), lambda s: s == "*") for t in tasks.values())

from __future__ import annotations

import pytest

from backend.app.agent.consolidated_tools import COVERS
from backend.app.agent.dev_tools.eval_tool_params import (
    PRESENT,
    canonicalize,
    load_labeled_set,
    score_trajectory,
)


@pytest.mark.parametrize(
    ("current", "consolidated"),
    [
        (("lookup_agency", {"name": "NSF"}), ("agency_profile", {"section": "overview", "agency_name": "NSF"})),
        (
            ("get_agency_budget", {"agency_name": "HHS", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "group_by": "sub_component"}),
            ("agency_profile", {"section": "budget_by_bureau", "agency_name": "HHS", "fiscal_year": 2024, "end_fiscal_year": 2024}),
        ),
        (
            ("get_agency_award_breakdown", {"agency_name": "NIH", "fiscal_year": 2024}),
            ("agency_profile", {"section": "awards_by_sub_agency", "agency_name": "NIH", "fiscal_year": 2024}),
        ),
        (("list_top_agencies_by_budget", {"limit": 5}), ("agency_profile", {"section": "rank_all_agencies", "limit": 5})),
        (("resolve_naics_code", {"description": "software"}), ("resolve_entity", {"kind": "naics", "query": "software"})),
        (("search_recipients", {"keyword": "Boeing"}), ("resolve_entity", {"kind": "recipient", "query": "Boeing"})),
        (("get_award_details", {"award_id": "X"}), ("award_detail", {"award_id": "X"})),
        (("get_award_subawards", {"award_id": "X"}), ("award_detail", {"award_id": "X", "section": "subawards"})),
        (("get_recipient_children", {"recipient_id": "R"}), ("recipient_profile", {"recipient_id": "R", "section": "children"})),
        (
            ("get_award_type_breakdown", {"start_year": 2024, "end_year": 2024}),
            ("query_spending", {"level": "award", "group_by": "award_type", "start_year": 2024, "end_year": 2024}),
        ),
    ],
)
def test_both_tool_sets_map_to_the_same_action(current, consolidated):
    current_action, current_args = canonicalize(*current)
    consolidated_action, consolidated_args = canonicalize(*consolidated)
    assert current_action == consolidated_action
    shared = set(current_args) & set(consolidated_args)
    assert {k: current_args[k] for k in shared} == {k: consolidated_args[k] for k in shared}


def test_every_current_tool_has_a_canonical_action():
    for consolidated_name, current_names in COVERS.items():
        for name in current_names:
            action, _ = canonicalize(name, {})
            assert action.split(".")[0] == consolidated_name, name


def _step(tool, args, output="ok"):
    return {"tool": tool, "args": args, "output": output}


def test_matching_call_passes():
    entry = {"expect": [{"action": "query_spending", "args": {"group_by": "naics", "agency_name": {"any_of": ["NSF", "National Science Foundation"]}}}]}
    result = score_trajectory([_step("query_spending", {"group_by": "naics", "agency_name": "national science foundation"})], entry)
    assert result["params_ok"] and result["action_ok"]


def test_wrong_param_fails_params_but_not_action():
    entry = {"expect": [{"action": "query_spending", "args": {"group_by": "naics"}}]}
    result = score_trajectory([_step("query_spending", {"group_by": "psc"})], entry)
    assert result["action_ok"] and not result["params_ok"]


def test_missing_action_fails_both():
    entry = {"expect": [{"action": "agency_profile.budget"}]}
    result = score_trajectory([_step("query_spending", {})], entry)
    assert not result["action_ok"] and not result["params_ok"]


def test_present_absent_every_call_and_forbid():
    entry = {
        "expect": [{"action": "query_spending", "args": {"recipient_id": PRESENT}, "absent": ["group_by"]}],
        "every_call": [{"action": "query_spending", "absent": ["award_type"]}],
        "forbid_actions": ["agency_profile.budget"],
    }
    good = [_step("query_spending", {"recipient_id": "R"})]
    assert score_trajectory(good, entry)["params_ok"]
    narrowed = [*good, _step("query_spending", {"recipient_id": "R", "award_type": "grants"})]
    assert not score_trajectory(narrowed, entry)["params_ok"]
    forbidden = [*good, _step("get_agency_budget", {"agency_name": "NSF"})]
    assert not score_trajectory(forbidden, entry)["params_ok"]


def test_rejected_call_fails_unless_errors_expected():
    trajectory = [
        _step("query_spending", {"group_by": "award_type", "award_type": "grants"}, "This query failed: award_type cannot be combined"),
        _step("query_spending", {"group_by": "naics"}),
    ]
    entry = {"expect": [{"action": "query_spending", "args": {"group_by": "naics"}}]}
    assert not score_trajectory(trajectory, entry)["params_ok"]
    assert score_trajectory(trajectory, {**entry, "expect_error": True})["params_ok"]


def test_one_of_accepts_any_alternative():
    entry = {"expect": [{"one_of": [
        {"action": "agency_profile.awards_by_sub_agency"},
        {"action": "query_spending", "args": {"group_by": "award_type"}},
    ]}]}
    assert score_trajectory([_step("query_spending", {"group_by": "award_type"})], entry)["params_ok"]
    assert score_trajectory([_step("get_agency_award_breakdown", {"agency_name": "NIH"})], entry)["params_ok"]
    assert not score_trajectory([_step("query_spending", {"group_by": "time"})], entry)["params_ok"]


def test_ordered_expectations():
    entry = {"ordered": True, "expect": [{"action": "query_spending"}, {"action": "award_detail.summary"}]}
    in_order = [_step("query_spending", {}), _step("get_award_details", {"award_id": "X"})]
    reversed_order = list(reversed(in_order))
    assert score_trajectory(in_order, entry)["params_ok"]
    assert not score_trajectory(reversed_order, entry)["params_ok"]


def test_labeled_set_loads_and_uses_known_actions():
    known = {canonicalize(name, {})[0] for names in COVERS.values() for name in names} | set(COVERS)
    known |= {f"{tool}.{section}" for tool in ("agency_profile", "award_detail", "recipient_profile") for section in (
        "overview", "budget", "budget_by_bureau", "awards_by_sub_agency", "awards_by_category", "award_activity",
        "rank_all_agencies", "summary", "transactions", "subawards", "funding", "details", "children",
    )}
    for entry in load_labeled_set().values():
        steps = [option for step in entry.get("expect", []) for option in step.get("one_of", [step])]
        for step in steps + entry.get("every_call", []):
            actions = step["action"] if isinstance(step["action"], list) else [step["action"]]
            assert set(actions) <= known, (entry["category"], actions)

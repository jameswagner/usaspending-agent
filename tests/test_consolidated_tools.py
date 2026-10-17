from __future__ import annotations

import json

import pytest

from backend.app.agent import consolidated_tools as ct
from backend.app.agent.langgraph_tools import _ARITHMETIC_TOOLS, LANGGRAPH_TOOLS
from backend.app.agent.orchestrator import _build_consolidated_system_prompt
from backend.app.agent.tools import MAX_TOOL_CALLS_PER_TURN, _tool_call_log

ARITHMETIC = {t.name for t in _ARITHMETIC_TOOLS}
TOOLS = {t.name: t for t in ct.CONSOLIDATED_TOOLS}
DOMAIN_TOOLS = [t for t in ct.CONSOLIDATED_TOOLS if t.name not in ARITHMETIC]


class _Recorder:
    def __init__(self, reply: str = "ok"):
        self.calls: list[dict] = []
        self.reply = reply

    def func(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


def _set_values(kwargs: dict) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


@pytest.fixture
def fake(monkeypatch):
    def install(name: str, reply: str = "ok") -> _Recorder:
        recorder = _Recorder(reply)
        monkeypatch.setattr(ct, name, recorder)
        return recorder

    return install


def test_every_current_tool_is_reachable_from_exactly_one_consolidated_tool():
    current = {t.name for t in LANGGRAPH_TOOLS} - ARITHMETIC
    covered = [name for names in ct.COVERS.values() for name in names]
    assert sorted(covered) == sorted(current)
    assert set(ct.COVERS) == {t.name for t in DOMAIN_TOOLS}


def test_no_schema_mentions_a_tool_missing_from_the_consolidated_set():
    for tool in ct.CONSOLIDATED_TOOLS:
        blob = tool.description + json.dumps(tool.tool_call_schema.model_json_schema())
        assert ct._RENAME_PATTERN.findall(blob) == [], tool.name


def test_prompt_names_every_consolidated_tool():
    prompt = _build_consolidated_system_prompt()
    for name in TOOLS:
        assert name in prompt, name


def test_budget_guard_covers_every_domain_tool():
    token = _tool_call_log.set([("lookup_agency", None, {})] * MAX_TOOL_CALLS_PER_TURN)
    try:
        for tool in DOMAIN_TOOLS:
            assert "Tool call budget exceeded" in tool.func(), tool.name
    finally:
        _tool_call_log.reset(token)


def test_results_have_stale_tool_names_rewritten(fake):
    fake("get_award_details", reply="Then call get_award_subawards for its subawards.")
    out = TOOLS["award_detail"].invoke({"award_id": "CONT_AWD_1"})
    assert out == 'Then call award_detail(section="subawards") for its subawards.'


@pytest.mark.parametrize(
    ("kind", "target", "expected"),
    [
        ("recipient", "search_recipients", {"keyword": "Boeing"}),
        ("naics", "resolve_naics_code", {"description": "Boeing"}),
        ("psc", "resolve_psc_code", {"description": "Boeing"}),
        ("cfda", "resolve_cfda_program", {"description": "Boeing"}),
        ("county", "resolve_county_fips", {"description": "Boeing"}),
        ("budget_function", "resolve_budget_function", {"description": "Boeing"}),
    ],
)
def test_resolve_entity_routes_by_kind(fake, kind, target, expected):
    recorder = fake(target)
    TOOLS["resolve_entity"].invoke({"kind": kind, "query": "Boeing"})
    assert recorder.calls == [expected]


def test_agency_profile_budget_defaults_end_year_to_start(fake):
    recorder = fake("get_agency_budget")
    TOOLS["agency_profile"].invoke({"section": "budget", "agency_name": "NSF", "fiscal_year": 2024})
    assert recorder.calls == [{
        "agency_name": "NSF", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "include_period_breakdown": False,
    }]


def test_agency_profile_budget_by_bureau_is_single_year(fake):
    recorder = fake("get_agency_budget")
    TOOLS["agency_profile"].invoke(
        {"section": "budget_by_bureau", "agency_name": "HHS", "fiscal_year": 2024, "bureau": "NIH"}
    )
    assert recorder.calls == [{
        "agency_name": "HHS", "start_fiscal_year": 2024, "end_fiscal_year": 2024,
        "group_by": "sub_component", "bureau": "NIH",
    }]


@pytest.mark.parametrize(
    ("section", "group_by"),
    [("awards_by_sub_agency", "sub_agency"), ("awards_by_category", "award_category"), ("award_activity", "whole_agency")],
)
def test_agency_profile_award_sections(fake, section, group_by):
    recorder = fake("get_agency_award_breakdown")
    TOOLS["agency_profile"].invoke({"section": section, "agency_name": "NSF", "fiscal_year": 2024})
    assert recorder.calls[0]["group_by"] == group_by


def test_agency_profile_overview_and_ranking(fake):
    overview = fake("lookup_agency")
    ranking = fake("list_top_agencies_by_budget")
    TOOLS["agency_profile"].invoke({"section": "overview", "agency_name": "NSF"})
    TOOLS["agency_profile"].invoke({"section": "rank_all_agencies", "limit": 3})
    assert overview.calls == [{"name": "NSF"}]
    assert ranking.calls == [{"limit": 3}]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"section": "budget", "agency_name": "NSF", "fiscal_year": 2024, "bureau": "X"}, "bureau does not apply"),
        ({"section": "rank_all_agencies", "agency_name": "NSF"}, "agency_name does not apply"),
        ({"section": "budget", "fiscal_year": 2024}, "agency_name is required"),
        ({"section": "awards_by_category", "agency_name": "NSF"}, "fiscal_year is required"),
    ],
)
def test_agency_profile_rejects_misplaced_or_missing_params(fake, args, message):
    fake("get_agency_budget")
    assert message in TOOLS["agency_profile"].invoke(args)


@pytest.mark.parametrize(
    ("section", "target"),
    [("transactions", "get_award_transaction_history"), ("subawards", "get_award_subawards"), ("funding", "get_award_funding_breakdown")],
)
def test_award_detail_sections(fake, section, target):
    recorder = fake(target)
    TOOLS["award_detail"].invoke({"award_id": "CONT_AWD_1", "section": section, "limit": 3})
    assert recorder.calls == [{"award_id": "CONT_AWD_1", "limit": 3}]


def test_award_detail_child_orders_only_for_summary(fake):
    fake("get_award_subawards")
    out = TOOLS["award_detail"].invoke({"award_id": "X", "section": "subawards", "include_child_orders": True})
    assert "include_child_orders does not apply" in out


def test_recipient_profile_sections(fake):
    details = fake("get_recipient_details")
    children = fake("get_recipient_children")
    TOOLS["recipient_profile"].invoke({"recipient_id": "abc-P"})
    TOOLS["recipient_profile"].invoke({"recipient_id": "abc-P", "section": "children", "year": "2023", "limit": 5})
    assert details.calls == [{"recipient_id": "abc-P", "year": "all"}]
    assert children.calls == [{"recipient_id": "abc-P", "year": "2023", "limit": 5}]


def test_query_spending_award_type_routes_to_the_count_breakdown(fake):
    counts = fake("get_award_type_breakdown")
    spending = fake("query_spending")
    TOOLS["query_spending"].invoke({
        "level": "award", "group_by": "award_type", "start_year": 2024, "end_year": 2024,
        "agency_name": "NSF", "limit": 5,
    })
    assert spending.calls == []
    assert len(counts.calls) == 1
    assert _set_values(counts.calls[0]) == {
        "time_period_type": "fiscal", "start_year": 2024, "end_year": 2024, "agency_name": "NSF",
    }


def test_query_spending_award_type_rejects_filters_the_breakdown_lacks(fake):
    counts = fake("get_award_type_breakdown")
    out = TOOLS["query_spending"].invoke({
        "level": "award", "group_by": "award_type", "start_year": 2024, "end_year": 2024, "award_type": "grants",
    })
    assert "award_type cannot be combined" in out
    assert counts.calls == []


def test_query_spending_other_group_bys_pass_through(fake):
    spending = fake("query_spending")
    TOOLS["query_spending"].invoke({"level": "award", "group_by": "naics", "start_year": 2024, "end_year": 2024})
    assert len(spending.calls) == 1
    assert {"level": "award", "group_by": "naics", "start_year": 2024, "end_year": 2024}.items() <= spending.calls[0].items()


def test_query_spending_rejects_unknown_arguments():
    out = TOOLS["query_spending"].invoke({"level": "award", "start_year": 2024, "end_year": 2024, "bogus": 1})
    assert out.startswith("This query failed: unsupported tool argument(s): bogus")


def test_kept_tools_keep_their_names():
    assert {"get_spending_explorer_breakdown", "get_disaster_spending_overview", "download_records"} <= set(TOOLS)

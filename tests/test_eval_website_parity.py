from __future__ import annotations

from backend.app.agent.dev_tools.check_website_coverage import (
    backend_endpoints,
    classify,
    load_tasks,
)
from backend.app.agent.dev_tools.eval_tool_params import check_references, extract
from backend.app.agent.dev_tools.eval_website_parity import (
    build_rows,
    load_cases,
    resolve_request,
)

KINDS = {"amount", "text", "date", "mentions"}


def test_every_chat_relevant_task_has_exactly_one_case():
    backend = backend_endpoints()
    relevant = {t["id"] for t in load_tasks() if classify(t, backend)[0] in {"covered", "local", "gap"}}
    case_ids = [c["task"] for c in load_cases()]
    assert len(case_ids) == len(set(case_ids))
    assert set(case_ids) == relevant


def test_every_case_is_well_formed():
    for case in load_cases():
        assert case["question"].strip()
        for answer in case["answers"]:
            assert answer["kind"] in KINDS, case["task"]
            assert ("any" in answer) if answer["kind"] == "mentions" else ("extract" in answer), case["task"]


def test_placeholders_are_filled_from_context():
    for row in build_rows(load_cases()):
        for ref in row["references"]:
            assert "{{" not in str(ref.get("request")), row["task"]


def test_resolve_request_fills_path_and_body():
    request = {"method": "POST", "path": "/api/v2/awards/{{id}}/", "body": {"award_id": "{{id}}"}}
    assert resolve_request(request, {"id": "X1"}) == {"method": "POST", "path": "/api/v2/awards/X1/", "body": {"award_id": "X1"}}


def test_extract_max_picks_the_largest_row():
    data = {"results": [{"name": "a", "amount": 1}, {"name": "b", "amount": "5"}, {"name": "c", "amount": None}]}
    assert extract(data, "results.max=amount.name") == "b"


def _check(answer, refs, response):
    cache = {}
    for ref in refs:
        if ref["kind"] != "mentions":
            ref["request"] = {"method": "GET", "path": "/x"}
    cache['{"method": "GET", "path": "/x"}'] = response
    return check_references(answer, {"reference": refs}, cache)


def test_date_kind_accepts_spelled_out_dates():
    refs = [{"label": "d", "kind": "date", "extract": "when"}]
    assert _check("It was certified on December 29, 2025.", refs, {"when": "2025-12-29T19:43:57Z"})["truth_ok"]
    assert not _check("It was certified in 2025.", refs, {"when": "2025-12-29T19:43:57Z"})["truth_ok"]


def test_mentions_kind_needs_no_replay():
    refs = [{"label": "m", "kind": "mentions", "any": ["lifetime"]}]
    assert check_references("That is the award's lifetime value.", {"reference": refs}, {})["truth_ok"]
    assert not check_references("That is FY2025 spending.", {"reference": refs}, {})["truth_ok"]


def test_long_text_matches_on_its_opening():
    refs = [{"label": "t", "kind": "text", "extract": "name"}]
    long_name = "Defense Environmental Cleanup, Environmental and Other Defense Activities, Energy"
    assert _check("The top account is Defense Environmental Cleanup, at $44B.", refs, {"name": long_name})["truth_ok"]


def test_string_amounts_are_compared_as_numbers():
    refs = [{"label": "a", "kind": "amount", "extract": "aggregate"}]
    assert _check("Outlays peaked at $7.67 billion.", refs, {"aggregate": "7666106608.57"})["truth_ok"]


def test_a_broken_reference_is_reported_not_raised():
    refs = [{"label": "a", "kind": "amount", "extract": "missing"}]
    result = _check("$5", refs, {"other": 1})
    assert not result["truth_ok"] and result["reference_errors"]

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.app.agent.dev_tools.eval_tool_selection import (
    _ordered_match_end,
    _trajectory_from_messages,
    tool_order_correct,
)


def _ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class TestTrajectoryFromMessages:
    def test_pairs_each_call_with_its_output_in_call_order(self):
        messages = [
            HumanMessage(content="how much did NSF spend on software in FY2023?"),
            _ai_tool_call("resolve_naics_code", {"description": "custom software development"}, "a"),
            ToolMessage(content="541511 - Custom Computer Programming Services", tool_call_id="a"),
            _ai_tool_call("search_awards", {"naics_code": "541511", "fiscal_year": 2023}, "b"),
            ToolMessage(content="12 awards found", tool_call_id="b"),
            AIMessage(content="NSF spent..."),
        ]

        assert _trajectory_from_messages(messages) == [
            {
                "tool": "resolve_naics_code",
                "args": {"description": "custom software development"},
                "output": "541511 - Custom Computer Programming Services",
            },
            {
                "tool": "search_awards",
                "args": {"naics_code": "541511", "fiscal_year": 2023},
                "output": "12 awards found",
            },
        ]

    def test_includes_a_tool_that_returned_its_failure_branch(self):
        """The case tool_citations can't represent - the tool returns before
        _record_tool_call, so it leaves no citation behind."""
        messages = [
            _ai_tool_call("get_agency_budget", {"agency_name": "Department of Atlantic Affairs"}, "a"),
            ToolMessage(
                content="This query failed: No agency found matching 'Department of Atlantic Affairs'.",
                tool_call_id="a",
            ),
        ]

        trajectory = _trajectory_from_messages(messages)

        assert [step["tool"] for step in trajectory] == ["get_agency_budget"]
        assert "No agency found matching" in trajectory[0]["output"]

    def test_keeps_repeated_calls_to_one_tool_separate(self):
        messages = [
            _ai_tool_call("get_agency_budget", {"agency_name": "NSF"}, "a"),
            ToolMessage(content="NSF budget", tool_call_id="a"),
            _ai_tool_call("get_agency_budget", {"agency_name": "NASA"}, "b"),
            ToolMessage(content="NASA budget", tool_call_id="b"),
        ]

        assert [step["args"]["agency_name"] for step in _trajectory_from_messages(messages)] == ["NSF", "NASA"]

    def test_no_tool_calls_gives_an_empty_trajectory(self):
        assert _trajectory_from_messages([HumanMessage(content="hi"), AIMessage(content="hello")]) == []

    def test_truncates_a_long_output(self):
        messages = [_ai_tool_call("search_awards", {}, "a"), ToolMessage(content="x" * 5000, tool_call_id="a")]

        assert len(_trajectory_from_messages(messages)[0]["output"]) == 2000


class TestOrderedMatchEnd:
    def test_returns_index_past_the_last_match(self):
        assert _ordered_match_end(["a", "b"], ["a", "b", "c"]) == 2

    def test_allows_unrelated_calls_in_between(self):
        assert _ordered_match_end(["a", "b"], ["a", "lookup_agency", "b"]) == 3

    def test_rejects_the_reverse_order(self):
        assert _ordered_match_end(["a", "b"], ["b", "a"]) is None

    def test_rejects_a_missing_tool(self):
        assert _ordered_match_end(["a", "b"], ["a"]) is None

    def test_matches_a_later_repeat_when_the_first_is_out_of_order(self):
        assert _ordered_match_end(["a", "b"], ["b", "a", "b"]) == 3


def _row(expected: dict, tools_called: list[str]):
    return SimpleNamespace(outputs={"tools_called": tools_called}), SimpleNamespace(outputs=expected)


class TestToolOrderCorrect:
    def test_not_applicable_without_expected_tools(self):
        run, example = _row({"expected_tool": "search_awards"}, ["search_awards"])

        assert tool_order_correct(run, example)["score"] is None

    def test_passes_when_the_chain_is_in_order(self):
        run, example = _row(
            {"expected_tools": ["search_awards", "get_award_details"]},
            ["search_awards", "get_award_details"],
        )

        assert tool_order_correct(run, example)["score"] == 1.0

    def test_fails_when_both_appear_but_reversed(self):
        """tool_selection_correct passes this one - membership is all it asks."""
        run, example = _row(
            {"expected_tools": ["resolve_naics_code", "search_awards"]},
            ["search_awards", "resolve_naics_code"],
        )

        assert tool_order_correct(run, example)["score"] == 0.0

    def test_then_one_of_must_follow_the_expected_tools(self):
        expected = {
            "expected_tools": ["resolve_naics_code"],
            "then_one_of": ["search_awards", "get_spending_by_category"],
        }

        after = tool_order_correct(*_row(expected, ["resolve_naics_code", "search_awards"]))
        before = tool_order_correct(*_row(expected, ["search_awards", "resolve_naics_code"]))

        assert after["score"] == 1.0
        assert before["score"] == 0.0

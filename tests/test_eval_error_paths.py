from types import SimpleNamespace
from unittest.mock import patch

from backend.app.agent.dev_tools import eval_tool_selection
from backend.app.agent.dev_tools.calibrate_failure_judge import (
    load_labeled_set as load_judge_fixtures,
)
from backend.app.agent.dev_tools.eval_tool_selection import (
    failure_acknowledged,
    load_labeled_set,
    tool_failure_observed,
)


def _row(trajectory: list[dict], answer: str = "", flagged: bool = True):
    outputs = {"expect_failure_acknowledged": True} if flagged else {"expected_tool": "search_awards"}
    return (
        SimpleNamespace(outputs={"trajectory": trajectory, "answer": answer}),
        SimpleNamespace(outputs=outputs, inputs={"question": "q"}),
    )


def _step(tool: str, output: str):
    return {"tool": tool, "args": {}, "output": output}


class TestToolFailureObserved:
    def test_not_applicable_when_the_case_is_not_an_error_path(self):
        run, example = _row([_step("get_agency_budget", "This query failed: nope.")], flagged=False)

        assert tool_failure_observed(run, example)["score"] is None

    def test_detects_the_query_failed_branch(self):
        run, example = _row([_step("get_agency_budget", "This query failed: No agency found matching 'x'.")])

        assert tool_failure_observed(run, example)["score"] == 1.0

    def test_detects_the_no_award_data_branch(self):
        run, example = _row([_step("get_agency_award_breakdown", "No award data found for EPA in FY2005.")])

        assert tool_failure_observed(run, example)["score"] == 1.0

    def test_flags_a_case_that_stopped_failing(self):
        """The live API changed and the 'broken' input now works - the case is
        stale and no longer tests anything."""
        run, example = _row([_step("get_agency_award_breakdown", "EPA issued 4,812 new awards.")])

        assert tool_failure_observed(run, example)["score"] == 0.0


class TestFailureAcknowledged:
    def test_not_applicable_when_no_tool_failed(self):
        run, example = _row([_step("get_agency_award_breakdown", "EPA issued 4,812 new awards.")])

        assert failure_acknowledged(run, example)["score"] is None

    def test_passes_the_failing_output_and_answer_to_the_judge(self):
        run, example = _row(
            [_step("get_agency_budget", "This query failed: No agency found matching 'x'.")],
            answer="I couldn't find that agency.",
        )

        with patch.object(eval_tool_selection, "judge_failure_acknowledged", return_value=(True, "YES - admits it")) as judge:
            result = failure_acknowledged(run, example)

        _question, tool_output, answer = judge.call_args[0]
        assert "No agency found matching" in tool_output
        assert answer == "I couldn't find that agency."
        assert result["score"] == 1.0

    def test_reports_a_negative_verdict(self):
        run, example = _row(
            [_step("get_agency_budget", "This query failed: No agency found matching 'x'.")],
            answer="The agency had $4.2 billion in FY2023.",
        )

        with patch.object(eval_tool_selection, "judge_failure_acknowledged", return_value=(False, "NO - invents a number")):
            result = failure_acknowledged(run, example)

        assert result["score"] == 0.0
        assert "invents a number" in result["comment"]


class TestLabeledSets:
    def test_every_error_path_case_is_flagged(self):
        cases = [e for e in load_labeled_set() if e["category"] == "error_paths"]

        assert cases
        assert all(e.get("expect_failure_acknowledged") for e in cases)

    def test_judge_fixtures_cover_both_labels(self):
        """A judge measured only on answers that should pass says nothing about
        whether it catches the ones that shouldn't."""
        fixtures = load_judge_fixtures()
        labels = [f["label"] for f in fixtures]

        assert any(labels) and not all(labels)
        assert all({"question", "tool_output", "answer", "label", "name"} <= set(f) for f in fixtures)

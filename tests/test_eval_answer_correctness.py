from types import SimpleNamespace
from unittest.mock import patch

from backend.app.agent.dev_tools import eval_tool_selection
from backend.app.agent.dev_tools.calibrate_answer_correctness import (
    load_labeled_set as load_judge_fixtures,
)
from backend.app.agent.dev_tools.eval_tool_selection import (
    answer_correctness_judged,
    answer_matches_tool_output,
    load_labeled_set,
)


def _row(trajectory: list[dict], answer: str = "", flagged: bool = True):
    outputs = {"check_answer_correctness": True} if flagged else {"expected_tool": "search_awards"}
    return (
        SimpleNamespace(outputs={"trajectory": trajectory, "answer": answer}),
        SimpleNamespace(outputs=outputs, inputs={"question": "q"}),
    )


def _step(tool: str, output: str):
    return {"tool": tool, "args": {}, "output": output}


class TestAnswerMatchesToolOutput:
    def test_not_applicable_when_the_case_is_not_flagged(self):
        run, example = _row([_step("search_awards", "Total: $4,200,000")], answer="$4.2 million", flagged=False)

        assert answer_matches_tool_output(run, example)["score"] is None

    def test_not_applicable_when_the_answer_states_no_figures(self):
        run, example = _row([_step("search_awards", "Total: $4,200,000")], answer="I couldn't find a clean total.")

        assert answer_matches_tool_output(run, example)["score"] is None

    def test_passes_when_every_figure_appears_in_the_tool_output(self):
        run, example = _row(
            [_step("search_awards", "NSF FY2023 total: $8,900,000,000. California: $2,225,000,000.")],
            answer="California received $2,225,000,000 of NSF's $8,900,000,000 FY2023 total.",
        )

        result = answer_matches_tool_output(run, example)
        assert result["score"] == 1.0

    def test_fails_when_a_figure_is_not_in_the_tool_output(self):
        run, example = _row(
            [_step("search_awards", "California: $612,000,000; Massachusetts: $398,000,000.")],
            answer="California led with $900,000,000 in NSF funding.",
        )

        result = answer_matches_tool_output(run, example)
        assert result["score"] == 0.0
        assert "900" in result["comment"]

    def test_tolerates_comma_and_dollar_sign_formatting_differences(self):
        run, example = _row(
            [_step("search_awards", "Total obligations: 4200000000")],
            answer="Total obligations were $4,200,000,000.",
        )

        assert answer_matches_tool_output(run, example)["score"] == 1.0

    def test_ignores_bare_four_digit_years(self):
        run, example = _row(
            [_step("search_awards", "NSF FY2023 total: $8,900,000,000.")],
            answer="In FY2023, NSF obligated $8,900,000,000.",
        )

        assert answer_matches_tool_output(run, example)["score"] == 1.0

    def test_matches_percentages(self):
        run, example = _row(
            [_step("search_awards", "California received 25% of the total.")],
            answer="California got about 25% of NSF's funding.",
        )

        assert answer_matches_tool_output(run, example)["score"] == 1.0


class TestAnswerCorrectnessJudged:
    def test_not_applicable_when_the_case_is_not_flagged(self):
        run, example = _row([_step("search_awards", "Total: $4,200,000")], flagged=False)

        assert answer_correctness_judged(run, example)["score"] is None

    def test_passes_the_trajectory_and_answer_to_the_judge(self):
        run, example = _row(
            [_step("search_awards", "NASA FY2024 grant total: $4,200,000,000.")],
            answer="NASA obligated $4.2 billion in FY2024 grants.",
        )

        with patch.object(
            eval_tool_selection, "judge_answer_correctness", return_value=(True, "matches")
        ) as judge:
            result = answer_correctness_judged(run, example)

        question, tool_output, answer = judge.call_args[0]
        assert question == "q"
        assert "4,200,000,000" in tool_output
        assert answer == "NASA obligated $4.2 billion in FY2024 grants."
        assert result["score"] == 1.0

    def test_reports_a_negative_verdict(self):
        run, example = _row(
            [_step("search_awards", "NASA FY2024 grant total: $4,200,000,000.")],
            answer="NASA obligated $4.2 million in FY2024 grants.",
        )

        with patch.object(
            eval_tool_selection, "judge_answer_correctness", return_value=(False, "wrong unit")
        ):
            result = answer_correctness_judged(run, example)

        assert result["score"] == 0.0
        assert "wrong unit" in result["comment"]


class TestLabeledSets:
    def test_check_answer_correctness_only_flags_the_six_spending_tools_categories(self):
        allowed_categories = {
            "category_breakdown",
            "time_breakdown",
            "geography_breakdown",
            "search_vs_breakdown",
            "subaward_tools",
            "transaction_tools",
            "followup_chains",
        }
        flagged = [e for e in load_labeled_set() if e.get("check_answer_correctness")]

        assert flagged
        assert all(e["category"] in allowed_categories for e in flagged)

    def test_judge_fixtures_cover_both_labels(self):
        """A judge measured only on answers that should pass says nothing about
        whether it catches the ones that shouldn't."""
        fixtures = load_judge_fixtures()
        labels = [f["label"] for f in fixtures]

        assert any(labels) and not all(labels)
        assert all({"question", "tool_output", "answer", "label", "name"} <= set(f) for f in fixtures)

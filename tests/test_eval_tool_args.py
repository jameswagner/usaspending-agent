import inspect
from types import SimpleNamespace

import pytest

from backend.app.agent.dev_tools.eval_tool_selection import (
    _arg_matches,
    load_labeled_set,
    tool_args_correct,
)
from backend.app.agent.langgraph_tools import _BETA_TOOLS


def _row(expected_args: dict, trajectory: list[dict]):
    return (
        SimpleNamespace(outputs={"trajectory": trajectory}),
        SimpleNamespace(outputs={"expected_args": expected_args}),
    )


def _step(tool: str, args: dict, output: str = ""):
    return {"tool": tool, "args": args, "output": output}


class TestArgMatches:
    def test_scalar_matches_across_int_and_string(self):
        assert _arg_matches(2023, "2023", {})
        assert _arg_matches("2023", 2023, {})

    def test_scalar_is_case_and_space_insensitive(self):
        assert _arg_matches("Department of Energy", " department of energy ", {})

    def test_scalar_mismatch(self):
        assert not _arg_matches(2023, 2024, {})

    def test_list_means_any_of(self):
        assert _arg_matches(["NIH", "National Institutes of Health"], "NIH", {})
        assert not _arg_matches(["NIH", "National Institutes of Health"], "NSF", {})

    def test_from_matcher_accepts_a_value_in_the_earlier_output(self):
        earlier = {"resolve_naics_code": ["541511 - Custom Computer Programming Services"]}

        assert _arg_matches("<from:resolve_naics_code>", "541511", earlier)

    def test_from_matcher_rejects_a_guessed_code(self):
        earlier = {"resolve_naics_code": ["541511 - Custom Computer Programming Services"]}

        assert not _arg_matches("<from:resolve_naics_code>", "999999", earlier)

    def test_from_matcher_rejects_when_the_source_never_ran(self):
        assert not _arg_matches("<from:resolve_naics_code>", "541511", {})


class TestToolArgsCorrect:
    def test_not_applicable_without_expected_args(self):
        run = SimpleNamespace(outputs={"trajectory": []})
        example = SimpleNamespace(outputs={"expected_tool": "search_awards"})

        assert tool_args_correct(run, example)["score"] is None

    def test_passes_when_args_match(self):
        run, example = _row(
            {"get_agency_award_breakdown": {"fiscal_year": 2023}},
            [_step("get_agency_award_breakdown", {"agency_name": "DOE", "fiscal_year": 2023})],
        )

        assert tool_args_correct(run, example)["score"] == 1.0

    def test_fails_on_the_wrong_year(self):
        run, example = _row(
            {"get_agency_award_breakdown": {"fiscal_year": 2023}},
            [_step("get_agency_award_breakdown", {"agency_name": "DOE", "fiscal_year": 2024})],
        )

        assert tool_args_correct(run, example)["score"] == 0.0

    def test_fails_when_the_argument_is_absent(self):
        run, example = _row(
            {"search_subawards": {"subrecipient_name": "Boeing"}},
            [_step("search_subawards", {"agency_name": "Boeing"})],
        )

        assert tool_args_correct(run, example)["score"] == 0.0

    def test_not_applicable_when_no_named_tool_was_called(self):
        """A then_one_of entry can name args for both branches; the branch that
        didn't run shouldn't count against the one that did."""
        run, example = _row(
            {"search_awards": {"naics_code": "541511"}},
            [_step("get_spending_by_category", {"naics_code": "541511"})],
        )

        assert tool_args_correct(run, example)["score"] is None

    def test_grades_only_the_branch_that_ran(self):
        run, example = _row(
            {
                "search_awards": {"naics_code": "<from:resolve_naics_code>"},
                "get_spending_by_category": {"naics_code": "<from:resolve_naics_code>"},
            },
            [
                _step("resolve_naics_code", {"description": "software"}, "541511 - Custom Computer Programming"),
                _step("get_spending_by_category", {"naics_code": "541511"}),
            ],
        )

        assert tool_args_correct(run, example)["score"] == 1.0

    def test_catches_a_code_the_model_guessed_instead_of_resolving(self):
        run, example = _row(
            {"search_awards": {"naics_code": "<from:resolve_naics_code>"}},
            [
                _step("resolve_naics_code", {"description": "software"}, "541511 - Custom Computer Programming"),
                _step("search_awards", {"naics_code": "511210"}),
            ],
        )

        assert tool_args_correct(run, example)["score"] == 0.0

    def test_a_later_correct_call_redeems_an_earlier_wrong_one(self):
        run, example = _row(
            {"get_agency_award_breakdown": {"fiscal_year": 2023}},
            [
                _step("get_agency_award_breakdown", {"fiscal_year": 2024}),
                _step("get_agency_award_breakdown", {"fiscal_year": 2023}),
            ],
        )

        assert tool_args_correct(run, example)["score"] == 1.0


def _seeded_args():
    for entry in load_labeled_set():
        for tool, args in (entry.get("expected_args") or {}).items():
            for name in args:
                yield entry["question"], tool, name


class TestSeededArgsAreReal:
    """expected_args names arguments by hand, so a renamed parameter would
    otherwise turn into a silently-always-failing eval case."""

    @pytest.mark.parametrize("question,tool,arg", list(_seeded_args()))
    def test_arg_exists_on_the_tool(self, question, tool, arg):
        by_name = {bt.name: bt.func for bt in _BETA_TOOLS}
        assert tool in by_name, f"{question!r} names unknown tool {tool!r}"
        assert arg in inspect.signature(by_name[tool]).parameters, (
            f"{question!r} expects {tool}.{arg}, which is not a parameter of {tool}"
        )

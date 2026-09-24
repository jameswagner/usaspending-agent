"""Opt-in coverage check: which tools have zero tool-selection eval cases?

Run before starting new tool work, not on a schedule:
    uv run python -m backend.app.agent.dev_tools.check_tool_eval_coverage

"All tools" comes from langgraph_tools._BETA_TOOLS rather than
tools/__init__.py's __all__, since __all__ also exports _raw variants,
_format_* helpers, and non-wired-up tools.

"Covered" is any tool name appearing under expected_tool, expected_tools,
acceptable_tools, then_one_of, or confusable_with anywhere in
tool_selection_labeled_set.json, or named as a key of expected_args - see that file's own "_notes" for what
each key means.
"""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.agent.langgraph_tools import _BETA_TOOLS

LABELED_SET_PATH = Path(__file__).parent / "tool_selection_labeled_set.json"

_COVERAGE_KEYS = (
    "expected_tool",
    "expected_tools",
    "acceptable_tools",
    "then_one_of",
    "confusable_with",
)


def _covered_tool_names(labeled_set: dict) -> set[str]:
    covered: set[str] = set()
    for category, cases in labeled_set.items():
        if category == "_notes":
            continue
        for case in cases:
            for key in _COVERAGE_KEYS:
                value = case.get(key)
                if value is None:
                    continue
                covered.update(value if isinstance(value, list) else [value])
            covered.update(case.get("expected_args", {}))
    return covered


def main() -> None:
    all_tools = {bt.name for bt in _BETA_TOOLS}
    labeled_set = json.loads(LABELED_SET_PATH.read_text())
    covered = _covered_tool_names(labeled_set)

    uncovered = sorted(all_tools - covered)
    print(f"{len(all_tools) - len(uncovered)}/{len(all_tools)} tools covered")

    if uncovered:
        print("\nUncovered tools (no eval case anywhere in the labeled set):")
        for name in uncovered:
            print(f"  - {name}")
    else:
        print("\nEvery tool has at least one eval case.")

    unknown = sorted(covered - all_tools)
    if unknown:
        print(
            f"\nFor reference, {len(unknown)} name(s) in the labeled set don't match "
            f"any current tool (renamed or removed?):"
        )
        for name in unknown:
            print(f"  - {name}")


if __name__ == "__main__":
    main()

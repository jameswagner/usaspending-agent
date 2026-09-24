"""Measure the answer-correctness judge against hand-labeled fixtures.

The judge is a classifier, so it has an accuracy, and until that accuracy is
measured its verdict stays diagnostic in eval_tool_selection.py rather than
gating a run. This is what produces the number that decides whether to flip it.

Cheap and repeatable: the fixtures are hand-written, so nothing here runs the
agent or calls the USASpending API - it's N_REPEATS judge calls per fixture and
nothing else.

Real, billed API calls. Not part of CI:
    uv run python -m backend.app.agent.dev_tools.calibrate_answer_correctness
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree, tracing_context

from backend.app.agent.dev_tools.answer_correctness_judge import (
    JUDGE_MODEL,
    judge_answer_correctness,
)

LABELED_SET_PATH = Path(__file__).parent / "answer_correctness_labeled_set.json"
N_REPEATS = 5


def load_labeled_set() -> list[dict]:
    data = json.loads(LABELED_SET_PATH.read_text(encoding="utf-8"))
    return [
        {**entry, "category": category}
        for category, entries in data.items()
        if category != "_notes"
        for entry in entries
    ]


def run_repeated(fn, n: int) -> list[tuple[bool, str]]:
    # A pool thread doesn't inherit the run tree, so each repeat would start its own trace.
    parent = get_current_run_tree()

    def call(_):
        with tracing_context(parent=parent):
            return fn()

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(call, range(n)))


@traceable(run_type="chain", name="calibrate_answer_correctness_entry")
def evaluate_entry(entry: dict) -> dict:
    results = run_repeated(
        lambda: judge_answer_correctness(entry["question"], entry["tool_output"], entry["answer"]),
        N_REPEATS,
    )
    verdicts = [verdict for verdict, _ in results]
    majority = sum(verdicts) > len(verdicts) / 2
    return {
        **entry,
        "majority": majority,
        "correct": majority == entry["label"],
        "agreement": max(sum(verdicts), len(verdicts) - sum(verdicts)) / len(verdicts),
        "reasons": [reason for _, reason in results],
    }


def print_report(results: list[dict]) -> None:
    correct = sum(r["correct"] for r in results)
    print(f"\nJudge model: {JUDGE_MODEL}, {N_REPEATS} repeats per fixture")
    print(f"Majority-vote accuracy: {correct}/{len(results)} ({correct / len(results):.0%})")

    unstable = [r for r in results if r["agreement"] < 1.0]
    print(f"\nFixtures where the repeats disagreed ({len(unstable)}/{len(results)}):")
    for r in unstable:
        print(f"  [{r['agreement']:.0%} agreement] {r['name']}")

    wrong = [r for r in results if not r["correct"]]
    print(f"\nWrong verdicts ({len(wrong)}/{len(results)}):")
    for r in wrong:
        print(f"  {r['name']}")
        print(f"    expected {r['label']}, judged {r['majority']}")
        print(f"    judge said: {r['reasons'][0]}")

    print("\nFalse verdicts by direction:")
    false_positive = [r for r in wrong if r["majority"] and not r["label"]]
    false_negative = [r for r in wrong if not r["majority"] and r["label"]]
    print(f"  too lenient (passed an incorrect answer):  {len(false_positive)}")
    print(f"  too strict (failed a correct answer):      {len(false_negative)}")


def main() -> None:
    entries = load_labeled_set()
    print(f"Loaded {len(entries)} fixtures. Running {len(entries) * N_REPEATS} judge calls...")
    results = [evaluate_entry(entry) for entry in entries]
    print_report(results)


if __name__ == "__main__":
    main()

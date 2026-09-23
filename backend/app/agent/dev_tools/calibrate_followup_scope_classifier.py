"""Calibrate the follow-up scope classifier (FOLLOWUP_SCOPE_CLASSIFIER_PROMPT)
against a hand-reviewed labeled set - closes issue #76, which tracked this
prompt shipping with zero measured accuracy while the bare-question path
already had calibrate_scope_classifier.py.

Calls the real _is_in_scope(question, recent_messages=...) - the exact
function the live app runs - rather than duplicating the prompt, since
(unlike the bare-question path) there's only one variant here to measure,
not several to compare.

recent_messages only needs the .type/.content shape _render_recent_exchanges
reads (see scope.py) - a real LangGraph message list isn't needed to
exercise this path.

Usage:
  uv run python -m backend.app.agent.dev_tools.calibrate_followup_scope_classifier
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.app.agent.scope import _is_in_scope
from backend.app.agent.singletons import warm_up

LABELED_SET_PATH = Path(__file__).parent / "followup_scope_classifier_labeled_set.json"
N_REPEATS = 5


class _FakeMessage:
    def __init__(self, type_: str, content: str) -> None:
        self.type = type_
        self.content = content


def load_labeled_set() -> list[dict]:
    with LABELED_SET_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    entries = []
    for category, items in data.items():
        if category == "_notes":
            continue
        for item in items:
            entries.append({**item, "category": category})
    return entries


def _build_history(prior_turns: list[dict]) -> list[_FakeMessage]:
    messages: list[_FakeMessage] = []
    for turn in prior_turns:
        messages.append(_FakeMessage("human", turn["question"]))
        messages.append(_FakeMessage("ai", turn["answer"]))
    return messages


def classify(new_question: str, prior_turns: list[dict]) -> bool:
    return _is_in_scope(new_question, recent_messages=_build_history(prior_turns))


def run_repeated(fn, n: int) -> list[bool]:
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda _: fn(), range(n)))


def evaluate_entry(entry: dict) -> dict:
    expected = entry["label"] == "in_scope"
    calls = run_repeated(lambda: classify(entry["new_question"], entry["prior_turns"]), N_REPEATS)
    majority = sum(calls) > len(calls) / 2
    return {
        "new_question": entry["new_question"],
        "category": entry["category"],
        "expected": expected,
        "calls": calls,
        "single_shot_accuracy": sum(1 for c in calls if c == expected) / len(calls),
        "majority_vote": majority,
        "majority_correct": majority == expected,
        "agreement_rate": max(sum(calls), len(calls) - sum(calls)) / len(calls),
    }


def print_report(results: list[dict]) -> None:
    print(f"\n{'category':<32} {'1shot acc':>10} {'majority acc':>13} {'n':>4}")
    print("-" * 65)

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    for category, rows in by_category.items():
        print(
            f"{category:<32} "
            f"{sum(row['single_shot_accuracy'] for row in rows) / len(rows):>10.1%} "
            f"{sum(row['majority_correct'] for row in rows) / len(rows):>13.1%} "
            f"{len(rows):>4}"
        )

    print("-" * 65)
    print(
        f"{'OVERALL':<32} "
        f"{sum(r['single_shot_accuracy'] for r in results) / len(results):>10.1%} "
        f"{sum(r['majority_correct'] for r in results) / len(results):>13.1%} "
        f"{len(results):>4}"
    )

    avg_agreement = sum(r["agreement_rate"] for r in results) / len(results)
    print(f"\nMean agreement rate across {N_REPEATS} repeats (flakiness indicator): {avg_agreement:.1%}")
    print("(100% = every repeat agreed; 60% is the floor for N=5, i.e. a 3/2 split)")

    print("\nQuestions where the classifier is flaky (not 100% agreement):")
    for r in results:
        if r["agreement_rate"] < 1.0:
            print(f"  [{r['category']}] {r['new_question']!r} - calls: {r['calls']}")

    print("\nQuestions where majority-vote is wrong:")
    for r in results:
        if not r["majority_correct"]:
            print(f"  [{r['category']}] {r['new_question']!r} - expected {r['expected']}, calls: {r['calls']}")


def main() -> None:
    warm_up()
    entries = load_labeled_set()
    print(f"Loaded {len(entries)} labeled follow-up questions. Running {N_REPEATS} repeats each "
          f"({len(entries) * N_REPEATS} total classifier calls)...")

    results = []
    for i, entry in enumerate(entries, start=1):
        print(f"  [{i}/{len(entries)}] {entry['new_question'][:70]!r}")
        results.append(evaluate_entry(entry))

    print_report(results)


if __name__ == "__main__":
    main()

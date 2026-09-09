"""Tool-selection eval (issue #20) - does the agent call the right data
tool, checked against tool_selection_labeled_set.json (schema documented
there) via a LangSmith Dataset + evaluate() experiment.

tools_called comes from AgentResult.tool_citations, not from reading
_tool_call_log after ask() returns (see red_team_jailbreak.py) - ask() is
@traceable, which isolates contextvars, so a set() inside it never
propagates back out; tool_citations is read inside ask() before return.

Real, billed API calls. Not part of CI:
    uv run python -m backend.app.agent.dev_tools.eval_tool_selection
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.schemas import Example, ExampleUpdate, Run

from backend.app.agent.orchestrator import ask
from backend.app.agent.singletons import warm_up

LABELED_SET_PATH = Path(__file__).parent / "tool_selection_labeled_set.json"
DATASET_NAME = "tool-selection-eval"

# Fixed so uuid5(NAMESPACE, question) is stable across runs, making re-sync an upsert.
_EXAMPLE_ID_NAMESPACE = uuid.UUID("6f6f3b0e-6e2a-4f0b-9a3d-6d1a6b1b6a10")


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


def _example_id(question: str) -> uuid.UUID:
    return uuid.uuid5(_EXAMPLE_ID_NAMESPACE, question)


def sync_dataset(client: Client, entries: list[dict]) -> str:
    """Create the dataset if needed, then create/update every entry by deterministic id."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME,
            description="Tool-selection eval for issue #20 - synced from "
            "tool_selection_labeled_set.json, do not hand-edit examples here.",
        )
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    existing_ids = {e.id for e in client.list_examples(dataset_id=dataset.id)}

    to_create, to_update = [], []
    for entry in entries:
        question = entry["question"]
        example_id = _example_id(question)
        outputs = {k: v for k, v in entry.items() if k != "question"}
        if example_id in existing_ids:
            to_update.append(ExampleUpdate(id=example_id, inputs={"question": question}, outputs=outputs))
        else:
            to_create.append({"id": example_id, "inputs": {"question": question}, "outputs": outputs})

    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)
    return dataset.id


def predict(inputs: dict) -> dict:
    result = ask(inputs["question"])
    return {
        "answer": result.answer_text,
        "tools_called": [tc.tool_name for tc in result.tool_citations],
    }


def _expected_description(expected: dict) -> str:
    if "expected_tools" in expected:
        return " -> ".join(expected["expected_tools"])
    if "acceptable_tools" in expected:
        return " or ".join(expected["acceptable_tools"])
    return expected["expected_tool"]


def tool_selection_correct(run: Run, example: Example) -> dict[str, Any]:
    tools_called = (run.outputs or {}).get("tools_called", [])
    expected = example.outputs or {}

    if "expected_tools" in expected:
        passed = all(t in tools_called for t in expected["expected_tools"])
    elif "acceptable_tools" in expected:
        passed = any(t in tools_called for t in expected["acceptable_tools"])
    else:
        passed = expected["expected_tool"] in tools_called

    return {
        "key": "tool_selection_correct",
        "score": float(passed),
        "comment": f"expected {_expected_description(expected)}, got {tools_called}",
    }


def confusable_alternative_called(run: Run, example: Example) -> dict[str, Any]:
    tools_called = (run.outputs or {}).get("tools_called", [])
    expected = example.outputs or {}
    confusable = expected.get("confusable_with", [])
    wrong = [t for t in confusable if t in tools_called]

    return {
        "key": "confusable_alternative_called",
        "score": float(bool(wrong)),
        "comment": f"also called: {wrong}" if wrong else "none",
    }


def _feedback_score(row: dict, key: str) -> float:
    for evaluation_result in row["evaluation_results"]["results"]:
        if evaluation_result.key == key:
            return evaluation_result.score
    return 0.0


def print_report(rows: list[dict]) -> None:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        category = row["example"].outputs["category"]
        by_category.setdefault(category, []).append(row)

    print("\nPass rate by category:")
    for category, category_rows in by_category.items():
        rate = sum(_feedback_score(r, "tool_selection_correct") for r in category_rows) / len(category_rows)
        print(f"  {category:<24} {rate:.1%} ({len(category_rows)})")
    overall = sum(_feedback_score(r, "tool_selection_correct") for r in rows) / len(rows)
    print(f"  {'OVERALL':<24} {overall:.1%} ({len(rows)})")

    confused = [r for r in rows if _feedback_score(r, "confusable_alternative_called") > 0]
    print(f"\nQuestions where a confusable alternative was also called ({len(confused)}/{len(rows)}):")
    for row in confused:
        question = row["example"].inputs["question"]
        category = row["example"].outputs["category"]
        print(f"  [{category}] {question!r}")

    failures = [r for r in rows if _feedback_score(r, "tool_selection_correct") == 0.0]
    print(f"\nFailing questions ({len(failures)}/{len(rows)}):")
    for row in failures:
        question = row["example"].inputs["question"]
        category = row["example"].outputs["category"]
        tools_called = (row["run"].outputs or {}).get("tools_called", [])
        print(f"  [{category}] {question!r} - tools called: {tools_called}")


def main() -> None:
    # Without this, evaluate()'s concurrency races the unlocked lazy singletons in singletons.py.
    warm_up()

    client = Client()
    entries = load_labeled_set()
    print(f"Syncing {len(entries)} labeled questions into LangSmith dataset {DATASET_NAME!r}...")
    sync_dataset(client, entries)

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[tool_selection_correct, confusable_alternative_called],
        experiment_prefix="tool-selection",
        client=client,
        # >1 hangs/spins CPU here even after warm_up() - a real, separate bug, not yet root-caused.
        max_concurrency=1,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View results: {results.url}")

    print_report(list(results))


if __name__ == "__main__":
    main()

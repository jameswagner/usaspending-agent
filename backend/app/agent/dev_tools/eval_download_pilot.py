"""Download-pilot eval (issue #238), same shape as eval_tool_selection.py. Real billed calls - not part of CI."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.schemas import Example, ExampleUpdate, Run

from backend.app.agent.orchestrator import NOT_FOUND_MESSAGE, ask
from backend.app.agent.singletons import warm_up

LABELED_SET_PATH = Path(__file__).parent / "download_labeled_set.json"
DATASET_NAME = "download-pilot-eval"

# Fixed so uuid5(NAMESPACE, question) is stable across runs, making re-sync an upsert.
_EXAMPLE_ID_NAMESPACE = uuid.UUID("2b7e6f2a-4c9d-4a9e-9c2b-1a9f7d8e6c31")


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
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME,
            description="Download-pilot eval for issue #238 - synced from "
            "download_labeled_set.json, do not hand-edit examples here.",
        )
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    existing_ids = {e.id for e in client.list_examples(dataset_id=dataset.id)}
    current_ids = {_example_id(entry["question"]) for entry in entries}

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

    orphaned = existing_ids - current_ids
    if orphaned:
        client.delete_examples(list(orphaned))

    return dataset.id


def predict(inputs: dict) -> dict:
    result = ask(inputs["question"])
    return {
        "answer": result.answer_text,
        "got_download": bool(result.downloads),
        "not_supported": "isn't supported yet" in result.answer_text,
        "fell_back_to_loop": not result.downloads and result.answer_text != NOT_FOUND_MESSAGE,
    }


def download_pilot_correct(run: Run, example: Example) -> dict[str, Any]:
    outputs = run.outputs or {}
    expected = example.outputs or {}

    if expected.get("expect_download"):
        passed = outputs.get("got_download", False)
    elif expected.get("expect_not_supported"):
        passed = outputs.get("not_supported", False)
    else:  # expect_fallback_to_loop
        passed = not outputs.get("got_download", False) and not outputs.get("not_supported", False)

    return {
        "key": "download_pilot_correct",
        "score": float(passed),
        "comment": f"expected {[k for k in expected if k.startswith('expect_')]}, got {outputs}",
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
        rate = sum(_feedback_score(r, "download_pilot_correct") for r in category_rows) / len(category_rows)
        print(f"  {category:<24} {rate:.1%} ({len(category_rows)})")
    overall = sum(_feedback_score(r, "download_pilot_correct") for r in rows) / len(rows)
    print(f"  {'OVERALL':<24} {overall:.1%} ({len(rows)})")

    failures = [r for r in rows if _feedback_score(r, "download_pilot_correct") == 0.0]
    print(f"\nFailing questions ({len(failures)}/{len(rows)}):")
    for row in failures:
        question = row["example"].inputs["question"]
        category = row["example"].outputs["category"]
        print(f"  [{category}] {question!r} - got: {row['run'].outputs}")


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
        evaluators=[download_pilot_correct],
        experiment_prefix="download-pilot",
        client=client,
        max_concurrency=1,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View results: {results.url}")

    rows = list(results)
    print_report(rows)


if __name__ == "__main__":
    main()

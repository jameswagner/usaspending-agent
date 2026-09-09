"""Retrieval-accuracy eval for resolve_naics_code/resolve_psc_code/
resolve_cfda_program (issue #51) - does the retriever itself return the
right code, independent of whether the model chooses to call the tool
(that's #20's separate concern).

Runs as a LangSmith Dataset + evaluate() experiment (same reasoning as
eval_tool_selection.py: LangSmith is already a live dependency here).
No LLM calls at all - this hits the local embedding/cross-encoder/
chroma/whoosh retrievers directly, so it's fast and free, but still
worth the same persisted/versioned treatment: issue #51's whole point is
that a re-tuned RERANK_CONFIDENCE_THRESHOLD or a re-ingested corpus
could silently regress this with nothing to notice.

code_lookup_labeled_set.json's schema is documented there. Grading:
  - expected_codes non-empty: pass if any of them is in the top-5
    results (top-5 recall - the eval's primary pass/fail). top-1 exact
    match is scored separately, informationally.
  - expected_codes == []: a genuine no-match case: pass if no candidate
    clears RERANK_CONFIDENCE_THRESHOLD, matching what the real
    resolve_*_code tool would say.

Run manually:
    uv run python -m backend.app.agent.dev_tools.eval_code_lookup
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langsmith import Client, evaluate
from langsmith.schemas import Example, ExampleUpdate, Run

from backend.app.agent.singletons import (
    RERANK_CONFIDENCE_THRESHOLD,
    _get_cfda_retriever,
    _get_naics_retriever,
    _get_psc_retriever,
)

LABELED_SET_PATH = Path(__file__).parent / "code_lookup_labeled_set.json"
DATASET_NAME = "code-lookup-eval"

_EXAMPLE_ID_NAMESPACE = uuid.UUID("a35c9f9a-3d2f-4c2a-9c8e-2f6e2b6b6a11")

_RETRIEVERS = {
    "naics": _get_naics_retriever,
    "psc": _get_psc_retriever,
    "cfda": _get_cfda_retriever,
}


def load_labeled_set() -> list[dict]:
    with LABELED_SET_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    entries = []
    for system, items in data.items():
        if system == "_notes":
            continue
        for item in items:
            entries.append({**item, "system": system})
    return entries


def _example_id(system: str, query: str) -> uuid.UUID:
    return uuid.uuid5(_EXAMPLE_ID_NAMESPACE, f"{system}:{query}")


def sync_dataset(client: Client, entries: list[dict]) -> str:
    """Create the dataset if needed, then create/update every entry by deterministic id, deleting any orphaned from a since-edited query."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        client.create_dataset(
            DATASET_NAME,
            description="Retrieval-accuracy eval for issue #51 - synced from "
            "code_lookup_labeled_set.json, do not hand-edit examples here.",
        )
    dataset = client.read_dataset(dataset_name=DATASET_NAME)
    existing_ids = {e.id for e in client.list_examples(dataset_id=dataset.id)}
    current_ids = {_example_id(entry["system"], entry["query"]) for entry in entries}

    to_create, to_update = [], []
    for entry in entries:
        example_id = _example_id(entry["system"], entry["query"])
        inputs = {"query": entry["query"], "system": entry["system"]}
        outputs = {"expected_codes": entry["expected_codes"], "system": entry["system"]}
        if example_id in existing_ids:
            to_update.append(ExampleUpdate(id=example_id, inputs=inputs, outputs=outputs))
        else:
            to_create.append({"id": example_id, "inputs": inputs, "outputs": outputs})

    if to_create:
        client.create_examples(dataset_id=dataset.id, examples=to_create)
    if to_update:
        client.update_examples(dataset_id=dataset.id, updates=to_update)

    orphaned = existing_ids - current_ids
    if orphaned:
        client.delete_examples(list(orphaned))

    return dataset.id


def predict(inputs: dict) -> dict:
    retriever = _RETRIEVERS[inputs["system"]]()
    results = retriever.retrieve(inputs["query"], top_k=5)
    return {
        "top_slugs": [r["slug"] for r in results],
        "top_scores": [r["rerank_score"] for r in results],
    }


def top5_recall(run: Run, example: Example) -> dict[str, Any]:
    expected = (example.outputs or {}).get("expected_codes", [])
    outputs = run.outputs or {}
    top_slugs = outputs.get("top_slugs", [])
    top_scores = outputs.get("top_scores", [])

    if not expected:
        passed = not any(s > RERANK_CONFIDENCE_THRESHOLD for s in top_scores)
        comment = "no-match case: " + ("correctly no confident candidate" if passed else f"a candidate cleared threshold: {top_slugs}")
    else:
        passed = any(slug in expected for slug in top_slugs)
        comment = f"expected one of {expected}, got {top_slugs}"

    return {"key": "top5_recall", "score": float(passed), "comment": comment}


def top1_exact(run: Run, example: Example) -> dict[str, Any]:
    expected = (example.outputs or {}).get("expected_codes", [])
    if not expected:
        return {"key": "top1_exact", "score": None, "comment": "not applicable - no-match case"}

    top_slugs = (run.outputs or {}).get("top_slugs", [])
    passed = bool(top_slugs) and top_slugs[0] in expected
    return {"key": "top1_exact", "score": float(passed), "comment": f"expected one of {expected}, got top-1 {top_slugs[:1]}"}


def print_report(rows: list[dict]) -> None:
    def score(row: dict, key: str) -> float | None:
        for r in row["evaluation_results"]["results"]:
            if r.key == key:
                return r.score
        return None

    by_system: dict[str, list[dict]] = {}
    for row in rows:
        by_system.setdefault(row["example"].outputs["system"], []).append(row)

    print(f"\n{'system':<10} {'top5 recall':>12} {'top1 exact':>12} {'n':>4}")
    print("-" * 42)
    for system, system_rows in by_system.items():
        recall = sum(score(r, "top5_recall") for r in system_rows) / len(system_rows)
        top1_scores = [score(r, "top1_exact") for r in system_rows if score(r, "top1_exact") is not None]
        top1 = sum(top1_scores) / len(top1_scores) if top1_scores else float("nan")
        print(f"{system:<10} {recall:>12.1%} {top1:>12.1%} {len(system_rows):>4}")

    failures = [r for r in rows if score(r, "top5_recall") == 0.0]
    print(f"\nFailing queries ({len(failures)}/{len(rows)}):")
    for row in failures:
        query = row["example"].inputs["query"]
        system = row["example"].outputs["system"]
        top_slugs = (row["run"].outputs or {}).get("top_slugs", [])
        print(f"  [{system}] {query!r} - top-5: {top_slugs}")


def main() -> None:
    # Pre-warm before concurrency below - same unlocked lazy-singleton race as eval_tool_selection.py.
    for get_retriever in _RETRIEVERS.values():
        get_retriever()

    client = Client()
    entries = load_labeled_set()
    print(f"Syncing {len(entries)} labeled queries into LangSmith dataset {DATASET_NAME!r}...")
    sync_dataset(client, entries)

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[top5_recall, top1_exact],
        experiment_prefix="code-lookup",
        client=client,
        # >1 segfaults here even after pre-warming - a real, separate concurrency bug, not yet root-caused.
        max_concurrency=1,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View results: {results.url}")

    print_report(list(results))


if __name__ == "__main__":
    main()

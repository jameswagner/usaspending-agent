"""Before/after eval for the #233 spending-by-agency-by-fiscal-year pilot
(backend/app/agent/spending_by_agency_fy_pilot.py) - checked against
spending_by_agency_fy_labeled_set.json via a LangSmith Dataset + evaluate()
experiment, same skeleton as eval_tool_selection.py.

Unlike eval_tool_selection.py, predict() calls ask() TWICE per question -
once with the pilot enabled (the "after" path) and once with it disabled via
SPENDING_BY_AGENCY_FY_PILOT_ENABLED=false (the "before" path, i.e. today's
tool-calling loop) - so a single run produces a real round-trip-count and
latency comparison per question, not just a pass/fail on whether the pilot
fired.

Real, billed API calls (2x per labeled question - both paths run). Not part
of CI:
    uv run python -m backend.app.agent.dev_tools.eval_spending_by_agency_fy_pilot
"""
from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

from langsmith import Client, evaluate
from langsmith.schemas import Example, ExampleUpdate, Run

from backend.app.agent import orchestrator
from backend.app.agent.orchestrator import ask
from backend.app.agent.singletons import warm_up

LABELED_SET_PATH = Path(__file__).parent / "spending_by_agency_fy_labeled_set.json"
DATASET_NAME = "spending-by-agency-fy-pilot-eval"

_EXAMPLE_ID_NAMESPACE = uuid.UUID("2b8f9b1e-8b9b-4e6b-9f0f-3b5a7f9c1d2e")

_PILOT_ENV_VAR = "SPENDING_BY_AGENCY_FY_PILOT_ENABLED"


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
            description="Before/after eval for issue #233 - synced from "
            "spending_by_agency_fy_labeled_set.json, do not hand-edit examples here.",
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


def _run_ask(question: str) -> tuple[Any, float]:
    conversation_id = str(uuid.uuid4())
    start = time.perf_counter()
    result = ask(question, conversation_id=conversation_id)
    latency_s = time.perf_counter() - start
    return result, latency_s


def predict(inputs: dict) -> dict:
    question = inputs["question"]

    fired = {"pilot": False}
    original_handler = orchestrator.handle_agency_fy_spending_request

    def _spy(q: str, cid: str):
        result = original_handler(q, cid)
        fired["pilot"] = result is not None
        return result

    with patch("backend.app.agent.orchestrator.handle_agency_fy_spending_request", side_effect=_spy):
        pilot_result, pilot_latency_s = _run_ask(question)

    os.environ[_PILOT_ENV_VAR] = "false"
    try:
        loop_result, loop_latency_s = _run_ask(question)
    finally:
        os.environ.pop(_PILOT_ENV_VAR, None)

    return {
        "pilot_fired": fired["pilot"],
        "pilot_answer": pilot_result.answer_text,
        "pilot_tools_called": [tc.tool_name for tc in pilot_result.tool_citations],
        "pilot_latency_s": pilot_latency_s,
        "loop_answer": loop_result.answer_text,
        "loop_tools_called": [tc.tool_name for tc in loop_result.tool_citations],
        "loop_latency_s": loop_latency_s,
    }


def pilot_fired_correctly(run: Run, example: Example) -> dict[str, Any]:
    expected = (example.outputs or {}).get("expect_pilot_fires", False)
    actual = (run.outputs or {}).get("pilot_fired", False)
    return {
        "key": "pilot_fired_correctly",
        "score": float(expected == actual),
        "comment": f"expected pilot_fires={expected}, got {actual}",
    }


def round_trip_reduced(run: Run, example: Example) -> dict[str, Any]:
    """Diagnostic only - only meaningful when the pilot actually fired
    (otherwise both paths ran the identical tool loop)."""
    outputs = run.outputs or {}
    if not outputs.get("pilot_fired"):
        return {"key": "round_trip_reduced", "score": None, "comment": "pilot did not fire"}
    pilot_calls = len(outputs.get("pilot_tools_called", []))
    loop_calls = len(outputs.get("loop_tools_called", []))
    return {
        "key": "round_trip_reduced",
        "score": float(pilot_calls <= loop_calls),
        "comment": f"pilot: {pilot_calls} tool call(s), loop: {loop_calls} tool call(s)",
    }


def _feedback_score(row: dict, key: str) -> float | None:
    for evaluation_result in row["evaluation_results"]["results"]:
        if evaluation_result.key == key:
            return evaluation_result.score
    return None


def print_report(rows: list[dict]) -> None:
    by_category: dict[str, list[dict]] = {}
    for row in rows:
        category = row["example"].outputs["category"]
        by_category.setdefault(category, []).append(row)

    print("\nPilot-fired-correctly rate by category:")
    for category, category_rows in by_category.items():
        rate = sum(_feedback_score(r, "pilot_fired_correctly") or 0.0 for r in category_rows) / len(category_rows)
        print(f"  {category:<24} {rate:.1%} ({len(category_rows)})")

    fired_rows = [r for r in rows if (r["run"].outputs or {}).get("pilot_fired")]
    print(f"\nBefore/after latency and round-trip count, questions where the pilot fired ({len(fired_rows)}/{len(rows)}):")
    for row in fired_rows:
        outputs = row["run"].outputs or {}
        question = row["example"].inputs["question"]
        pilot_calls = len(outputs.get("pilot_tools_called", []))
        loop_calls = len(outputs.get("loop_tools_called", []))
        pilot_s = outputs.get("pilot_latency_s", 0.0)
        loop_s = outputs.get("loop_latency_s", 0.0)
        print(
            f"  {question!r}\n"
            f"    tool calls: pilot={pilot_calls} loop={loop_calls}  "
            f"latency: pilot={pilot_s:.2f}s loop={loop_s:.2f}s (delta {loop_s - pilot_s:+.2f}s)"
        )
    if fired_rows:
        mean_pilot_s = sum((r["run"].outputs or {}).get("pilot_latency_s", 0.0) for r in fired_rows) / len(fired_rows)
        mean_loop_s = sum((r["run"].outputs or {}).get("loop_latency_s", 0.0) for r in fired_rows) / len(fired_rows)
        print(f"\n  Mean latency: pilot={mean_pilot_s:.2f}s loop={mean_loop_s:.2f}s (delta {mean_loop_s - mean_pilot_s:+.2f}s)")

    failures = [r for r in rows if (_feedback_score(r, "pilot_fired_correctly") or 0.0) == 0.0]
    print(f"\nMisfires ({len(failures)}/{len(rows)}):")
    for row in failures:
        question = row["example"].inputs["question"]
        expected = row["example"].outputs.get("expect_pilot_fires")
        actual = (row["run"].outputs or {}).get("pilot_fired")
        print(f"  {question!r} - expected pilot_fires={expected}, got {actual}")


def main() -> None:
    warm_up()

    client = Client()
    entries = load_labeled_set()
    print(f"Syncing {len(entries)} labeled questions into LangSmith dataset {DATASET_NAME!r}...")
    sync_dataset(client, entries)

    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[pilot_fired_correctly, round_trip_reduced],
        experiment_prefix="spending-by-agency-fy-pilot",
        client=client,
        max_concurrency=1,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View results: {results.url}")

    rows = list(results)
    print_report(rows)


if __name__ == "__main__":
    main()

"""Tool-selection eval (issue #20) - does the agent call the right data
tool, checked against tool_selection_labeled_set.json (schema documented
there) via a LangSmith Dataset + evaluate() experiment.

tools_called comes from the LangGraph checkpointer's persisted messages for
the run's thread - tool_citations drops search_guide, error-branch returns and
repeat calls, and carries display parameters rather than the call's arguments.

Real, billed API calls. Not part of CI:
    uv run python -m backend.app.agent.dev_tools.eval_tool_selection
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langsmith import Client, evaluate
from langsmith.schemas import Example, ExampleUpdate, Run

from backend.app.agent.orchestrator import ask
from backend.app.agent.singletons import _get_conversation_graph, warm_up

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

    # A question's id is derived from its text (_example_id), so editing a
    # question's wording orphans its old example under the old id - delete
    # whatever's left in the dataset that isn't in the current JSON.
    orphaned = existing_ids - current_ids
    if orphaned:
        client.delete_examples(list(orphaned))

    return dataset.id


_TOOL_OUTPUT_CHARS = 2000


def _trajectory_from_messages(messages: list) -> list[dict]:
    """Every tool call in the thread, in call order, with its arguments and output."""
    outputs = {m.tool_call_id: m.content for m in messages if isinstance(m, ToolMessage)}
    return [
        {
            "tool": call["name"],
            "args": call["args"],
            "output": str(outputs.get(call["id"], ""))[:_TOOL_OUTPUT_CHARS],
        }
        for message in messages
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    ]


def predict(inputs: dict) -> dict:
    result = ask(inputs["question"])
    config = {"configurable": {"thread_id": result.conversation_id}}
    trajectory = _trajectory_from_messages(_get_conversation_graph().get_state(config).values.get("messages", []))
    if not trajectory:
        # A scope-gate rejection and the pilots answer without entering the graph.
        trajectory = [{"tool": tc.tool_name, "args": {}, "output": ""} for tc in result.tool_citations]
    return {
        "answer": result.answer_text,
        "tools_called": [step["tool"] for step in trajectory],
        "trajectory": trajectory,
    }


def _expected_description(expected: dict) -> str:
    if "expected_tools" in expected:
        desc = " -> ".join(expected["expected_tools"])
        if expected.get("then_one_of"):
            desc += " -> (one of) " + " or ".join(expected["then_one_of"])
        return desc
    if "acceptable_tools" in expected:
        return " or ".join(expected["acceptable_tools"])
    return expected["expected_tool"]


def tool_selection_correct(run: Run, example: Example) -> dict[str, Any]:
    tools_called = (run.outputs or {}).get("tools_called", [])
    expected = example.outputs or {}

    if "expected_tools" in expected:
        passed = all(t in tools_called for t in expected["expected_tools"])
        # then_one_of: a second step where more than one tool is a
        # legitimate choice (e.g. resolve_naics_code -> either
        # search_awards or get_spending_by_category both correctly use
        # the resolved code) - pinning one specific tool here mislabels
        # the other as a failure. AND'd onto expected_tools, not a
        # replacement for it.
        if passed and expected.get("then_one_of"):
            passed = any(t in tools_called for t in expected["then_one_of"])
    elif "acceptable_tools" in expected:
        passed = any(t in tools_called for t in expected["acceptable_tools"])
    else:
        passed = expected["expected_tool"] in tools_called

    return {
        "key": "tool_selection_correct",
        "score": float(passed),
        "comment": f"expected {_expected_description(expected)}, got {tools_called}",
    }


def _ordered_match_end(expected: list[str], tools_called: list[str]) -> int | None:
    """Index past the last expected tool, or None if it isn't an ordered
    subsequence. Unrelated calls in between are fine."""
    position = 0
    for tool in expected:
        try:
            position = tools_called.index(tool, position) + 1
        except ValueError:
            return None
    return position


def tool_order_correct(run: Run, example: Example) -> dict[str, Any]:
    """Order, which tool_selection_correct doesn't check - it passes as long as
    both tools appear anywhere. Its own key, so that metric stays comparable."""
    expected = example.outputs or {}
    if "expected_tools" not in expected:
        return {"key": "tool_order_correct", "score": None, "comment": "not applicable"}

    tools_called = (run.outputs or {}).get("tools_called", [])
    match_end = _ordered_match_end(expected["expected_tools"], tools_called)
    passed = match_end is not None
    if passed and expected.get("then_one_of"):
        passed = any(t in tools_called[match_end:] for t in expected["then_one_of"])

    return {
        "key": "tool_order_correct",
        "score": float(passed),
        "comment": f"expected {_expected_description(expected)} in that order, got {tools_called}",
    }


_FROM_PREFIX = "<from:"


def _scalar_matches(expected: Any, actual: Any) -> bool:
    return str(expected).strip().lower() == str(actual).strip().lower()


def _arg_matches(expected: Any, actual: Any, earlier_outputs: dict[str, list[str]]) -> bool:
    if isinstance(expected, str) and expected.startswith(_FROM_PREFIX):
        source = expected[len(_FROM_PREFIX) :].rstrip(">")
        return any(str(actual) in output for output in earlier_outputs.get(source, []))
    if isinstance(expected, list):
        return any(_scalar_matches(option, actual) for option in expected)
    return _scalar_matches(expected, actual)


def _call_matches(expected_args: dict, args: dict, earlier_outputs: dict[str, list[str]]) -> bool:
    return all(
        name in args and _arg_matches(value, args[name], earlier_outputs)
        for name, value in expected_args.items()
    )


def tool_args_correct(run: Run, example: Example) -> dict[str, Any]:
    """Were the arguments right, not just the tool name. Only grades tools the run
    actually called - whether it called them at all is tool_selection_correct's job,
    and that split is what lets a then_one_of case name args for either branch."""
    expected = (example.outputs or {}).get("expected_args")
    if not expected:
        return {"key": "tool_args_correct", "score": None, "comment": "not applicable"}

    trajectory = (run.outputs or {}).get("trajectory", [])
    earlier_outputs: dict[str, list[str]] = {}
    graded: dict[str, bool] = {}
    for step in trajectory:
        tool = step.get("tool")
        if tool in expected and not graded.get(tool):
            graded[tool] = _call_matches(expected[tool], step.get("args") or {}, earlier_outputs)
        earlier_outputs.setdefault(tool, []).append(step.get("output") or "")

    if not graded:
        return {"key": "tool_args_correct", "score": None, "comment": "none of the named tools were called"}

    wrong = [tool for tool, ok in graded.items() if not ok]
    comment = "all matched" if not wrong else "; ".join(
        f"{tool}: expected {expected[tool]}, got "
        f"{[s.get('args') for s in trajectory if s.get('tool') == tool]}"
        for tool in wrong
    )
    return {"key": "tool_args_correct", "score": float(not wrong), "comment": comment[:500]}


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


_HEDGE_PHRASES = (
    "semantic match", "closest match", "not confirmed", "not a confirmed",
    "verify", "confirming", "approximate", "may not be exact",
)


def hedge_language_present(run: Run, example: Example) -> dict[str, Any]:
    """Diagnostic only (issue #52's 4th ask) - not gated into tool_selection_correct,
    same reasoning as red_team_jailbreak.py's own note on keyword checks: a
    hedge-phrase match is suggestive, not proof of genuine hedging, so this
    is reported for a human to read, not treated as a strict pass/fail."""
    expected = example.outputs or {}
    if not expected.get("check_hedge_language"):
        return {"key": "hedge_language_present", "score": None, "comment": "not applicable"}

    answer = ((run.outputs or {}).get("answer") or "").lower()
    passed = any(phrase in answer for phrase in _HEDGE_PHRASES)
    return {"key": "hedge_language_present", "score": float(passed), "comment": answer[:200]}


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

    ordered = [r for r in rows if _feedback_score(r, "tool_order_correct") is not None]
    if ordered:
        rate = sum(_feedback_score(r, "tool_order_correct") for r in ordered) / len(ordered)
        print(f"\nTool-order pass rate, multi-step entries only: {rate:.1%} ({len(ordered)})")
        for row in ordered:
            if _feedback_score(row, "tool_order_correct") > 0:
                continue
            question = row["example"].inputs["question"]
            membership = "membership passed" if _feedback_score(row, "tool_selection_correct") > 0 else "membership failed too"
            tools_called = (row["run"].outputs or {}).get("tools_called", [])
            print(f"  [{membership}] {question!r} - tools called: {tools_called}")

    arg_graded = [r for r in rows if _feedback_score(r, "tool_args_correct") is not None]
    if arg_graded:
        rate = sum(_feedback_score(r, "tool_args_correct") for r in arg_graded) / len(arg_graded)
        print(f"\nTool-argument pass rate, entries with expected_args: {rate:.1%} ({len(arg_graded)})")
        for row in arg_graded:
            if _feedback_score(row, "tool_args_correct") > 0:
                continue
            print(f"  {row['example'].inputs['question']!r}")

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

    hedge_checked = [r for r in rows if _feedback_score(r, "hedge_language_present") is not None]
    print(f"\nHedge-language check (diagnostic only, {len(hedge_checked)} entries flagged for it):")
    for row in hedge_checked:
        question = row["example"].inputs["question"]
        hedged = _feedback_score(row, "hedge_language_present") > 0
        print(f"  {'[hedged]' if hedged else '[NOT hedged]'} {question!r}")


def dump_trajectories(rows: list[dict], path: Path) -> None:
    """Per-question trajectory (tool, args, output), keyed by question."""
    data = {row["example"].inputs["question"]: (row["run"].outputs or {}).get("trajectory", []) for row in rows}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nDumped per-question trajectories to {path}")


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
        evaluators=[
            tool_selection_correct,
            tool_order_correct,
            tool_args_correct,
            confusable_alternative_called,
            hedge_language_present,
        ],
        experiment_prefix="tool-selection",
        client=client,
        # >1 hangs/spins CPU here even after warm_up() - a real, separate bug, not yet root-caused.
        max_concurrency=1,
    )

    print(f"\nExperiment: {results.experiment_name}")
    print(f"View results: {results.url}")

    rows = list(results)
    print_report(rows)
    dump_trajectories(rows, Path("/tmp/tool_selection.json"))


if __name__ == "__main__":
    main()

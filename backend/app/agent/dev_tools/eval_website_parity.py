"""Website parity eval: asks the agent each chat-relevant task from the usaspending.gov
audit and checks its answer against the website's own request, replayed live.

Asking the agent makes real, billed API calls. Not part of CI:
    uv run python -m backend.app.agent.dev_tools.eval_website_parity --toolsets current consolidated
    uv run python -m backend.app.agent.dev_tools.eval_website_parity --tasks Q001 Q102 --toolsets current
Free modes:
    uv run python -m backend.app.agent.dev_tools.eval_website_parity --check-references
    uv run python -m backend.app.agent.dev_tools.eval_website_parity --rescore data/website_parity/parity-<stamp>.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from backend.app.agent.dev_tools.check_website_coverage import (
    INVENTORY_PATH,
    backend_endpoints,
    classify,
    load_tasks,
)
from backend.app.agent.dev_tools.eval_tool_params import (
    _replayed_value,
    check_references,
)

CASES_PATH = Path(__file__).parent / "website_parity_cases.json"
CONTEXT_PATH = INVENTORY_PATH.parent / "context.json"
OUTPUT_DIR = Path("data/website_parity")
REACHABLE = {"covered", "local"}


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def resolve_request(request: dict, context: dict) -> dict:
    text = json.dumps(request)
    for key, value in context.items():
        text = text.replace("{{" + key + "}}", str(value))
    return json.loads(text)


def references_for(case: dict, task: dict, context: dict) -> list[dict]:
    refs = []
    for answer in case["answers"]:
        ref = dict(answer)
        if ref["kind"] != "mentions":
            ref["request"] = resolve_request(ref.get("request") or task["request"], context)
        refs.append(ref)
    return refs


def _ask(question: str, toolset: str) -> dict:
    from backend.app.agent.dev_tools.compare_raw_api import _run_wrapper_arm

    try:
        return _run_wrapper_arm(question, toolset)
    except Exception as exc:  # noqa: BLE001 - one crashed case shouldn't sink the whole run
        return {"error": f"{type(exc).__name__}: {exc}", "answer": ""}


def build_rows(cases: list[dict]) -> list[dict]:
    tasks = {t["id"]: t for t in load_tasks()}
    context = json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    backend = backend_endpoints()
    rows = []
    for case in cases:
        task = tasks[case["task"]]
        rows.append({
            "task": case["task"],
            "section": task["section"],
            "status": classify(task, backend)[0],
            "question": case["question"],
            "confusion": case.get("confusion"),
            "references": references_for(case, task, context),
        })
    return rows


def score_rows(rows: list[dict], toolsets: list[str]) -> None:
    cache: dict[str, dict] = {}
    for row in rows:
        for toolset in toolsets:
            if toolset in row:
                row[toolset].update(check_references(row[toolset].get("answer") or "", {"reference": row["references"]}, cache))


def check_reference_values(rows: list[dict]) -> str:
    cache: dict[str, dict] = {}
    lines, broken = [], 0
    for row in rows:
        for ref in row["references"]:
            if ref["kind"] == "mentions":
                continue
            try:
                value = _replayed_value(ref, cache)
                lines.append(f"{row['task']} {ref['label']}: {str(value)[:80]}")
            except Exception as exc:  # noqa: BLE001 - report every broken reference, not just the first
                broken += 1
                lines.append(f"{row['task']} {ref['label']}: BROKEN {type(exc).__name__}: {exc}")
    return "\n".join(lines + ["", f"{broken} broken reference(s)"])


def _rate(rows: list[dict], toolset: str) -> str:
    scored = [r for r in rows if "truth_ok" in r.get(toolset, {})]
    right = sum(r[toolset]["truth_ok"] for r in scored)
    return f"{right}/{len(scored)}" if scored else "-"


def summarize(rows: list[dict], toolsets: list[str]) -> str:
    reachable = [r for r in rows if r["status"] in REACHABLE]
    gaps = [r for r in rows if r["status"] == "gap"]
    lines = [
        "| | " + " | ".join(toolsets) + " |",
        "|---" * (len(toolsets) + 1) + "|",
        "| all chat-relevant tasks | " + " | ".join(_rate(rows, t) for t in toolsets) + " |",
        "| reachable (backend calls the endpoint) | " + " | ".join(_rate(reachable, t) for t in toolsets) + " |",
        "| gaps (no endpoint yet) | " + " | ".join(_rate(gaps, t) for t in toolsets) + " |",
        "| mean latency (s) | " + " | ".join(_mean_latency(rows, t) for t in toolsets) + " |",
        "",
        "| section | " + " | ".join(toolsets) + " |",
        "|---" * (len(toolsets) + 1) + "|",
    ]
    by_section: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_section[row["section"]].append(row)
    for section, section_rows in sorted(by_section.items()):
        lines.append(f"| {section} | " + " | ".join(_rate(section_rows, t) for t in toolsets) + " |")
    lines += ["", "## Wrong answers on reachable tasks", ""]
    for row in reachable:
        for toolset in toolsets:
            result = row.get(toolset, {})
            if result.get("truth_ok") is False:
                why = "; ".join(result.get("truth_problems", []) + result.get("reference_errors", []) + ([result["error"]] if result.get("error") else []))
                lines.append(f"- {row['task']} [{toolset}] {row['question']} - {why}")
    return "\n".join(lines)


def _mean_latency(rows: list[dict], toolset: str) -> str:
    values = [r[toolset]["latency_s"] for r in rows if "latency_s" in r.get(toolset, {})]
    return f"{sum(values) / len(values):.1f}" if values else "-"


def _write(rows: list[dict], toolsets: list[str], stamp: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"parity-{stamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    summary = summarize(rows, toolsets)
    out_path.with_suffix(".md").write_text(summary + "\n", encoding="utf-8")
    print("\n" + summary + f"\n\nRows: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--toolsets", nargs="+", default=["current", "consolidated"], choices=["current", "consolidated"])
    parser.add_argument("--tasks", nargs="*", help="only these audit task ids")
    parser.add_argument("--limit", type=int, help="first N cases")
    parser.add_argument("--check-references", action="store_true", help="replay every reference and print its value; no agent")
    parser.add_argument("--rescore", type=Path, help="re-score a saved parity-<stamp>.jsonl without asking the agent again")
    args = parser.parse_args()

    if args.rescore:
        rows = [json.loads(line) for line in args.rescore.read_text(encoding="utf-8").splitlines() if line.strip()]
        toolsets = [t for t in ("current", "consolidated") if any(t in r for r in rows)]
        score_rows(rows, toolsets)
        _write(rows, toolsets, args.rescore.stem.removeprefix("parity-") + "-rescored")
        return

    cases = load_cases()
    if args.tasks:
        cases = [c for c in cases if c["task"] in set(args.tasks)]
    if args.limit:
        cases = cases[: args.limit]
    rows = build_rows(cases)
    if args.check_references:
        print(check_reference_values(rows))
        return

    from backend.app.agent.singletons import warm_up

    warm_up()
    for row in rows:
        for toolset in args.toolsets:
            row[toolset] = _ask(row["question"], toolset)
        print(f"{row['task']} {row['question'][:90]}", flush=True)
    score_rows(rows, args.toolsets)
    _write(rows, args.toolsets, datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))


if __name__ == "__main__":
    main()

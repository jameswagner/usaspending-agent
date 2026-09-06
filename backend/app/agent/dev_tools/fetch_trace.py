"""Fetch one LangSmith trace by ID and dump it (and every run in it) to JSON.

Not a new dependency - python-dotenv and langsmith are both already
project dependencies. This is just a standalone script since it's a
one-off diagnostic, not something ask() ever needs at runtime.

Requires LANGSMITH_API_KEY in .env to have READ access (a Personal Access
Token, not an ingestion/service-only key) - the same key that can write
traces (tracing already works) may not be able to read them back; that's
a LangSmith project settings issue, not something this script can work
around.

Usage:
  uv run python -m backend.app.agent.dev_tools.fetch_trace <trace_or_run_id> [output.json]
"""
from __future__ import annotations

import json
import sys

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()


def fetch_trace(trace_id: str) -> list[dict]:
    client = Client()

    # trace_id here can be any run id that belongs to the trace - LangSmith
    # resolves it to the whole tree via list_runs(trace_id=...).
    runs = list(client.list_runs(trace_id=trace_id))
    runs.sort(key=lambda r: r.start_time)

    return [
        {
            "id": str(r.id),
            "parent_run_id": str(r.parent_run_id) if r.parent_run_id else None,
            "name": r.name,
            "run_type": r.run_type,
            "start_time": r.start_time.isoformat() if r.start_time else None,
            "end_time": r.end_time.isoformat() if r.end_time else None,
            "latency_s": (r.end_time - r.start_time).total_seconds() if r.end_time and r.start_time else None,
            "inputs": r.inputs,
            "outputs": r.outputs,
            "error": r.error,
        }
        for r in runs
    ]


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: fetch_trace.py <trace_or_run_id> [output.json]")
        sys.exit(1)

    trace_id = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else "trace.json"

    runs = fetch_trace(trace_id)
    with open(out_path, "w") as f:
        json.dump(runs, f, indent=2, default=str)

    print(f"Wrote {len(runs)} run(s) to {out_path}")


if __name__ == "__main__":
    main()

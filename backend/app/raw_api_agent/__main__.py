"""uv run python -m backend.app.raw_api_agent sync
uv run python -m backend.app.raw_api_agent ask "How much did NASA obligate in FY2024?"
"""
from __future__ import annotations

import argparse
import json

from .contracts import sync_contracts


def main() -> None:
    parser = argparse.ArgumentParser(prog="raw_api_agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sync", help="fetch upstream api_contracts into data/api_contracts")
    ask_parser = sub.add_parser("ask", help="answer one question")
    ask_parser.add_argument("question")
    ask_parser.add_argument("--trace", action="store_true", help="print every tool call")
    args = parser.parse_args()

    if args.command == "sync":
        print(f"Synced api_contracts at upstream commit {sync_contracts()}")
        return

    from .agent import ask

    result = ask(args.question)
    if args.trace:
        for step in result.trajectory:
            print(f"--> {step['tool']}({json.dumps(step['args'])[:400]})")
            print(f"    {step['output'][:300]!r}")
    print(result.answer)
    print(f"\n[{result.model}, {result.iterations} model calls, {len(result.api_calls)} API calls, "
          f"{result.latency_s}s, usage {result.usage}]")


if __name__ == "__main__":
    main()

"""CLI entry point.

Usage:
  python -m backend.app.agent --question "What is a prime award?"
"""
from __future__ import annotations

from backend.app.logging_config import configure_logging

from .orchestrator import ask
from .singletons import MODEL, warm_up


def main():
    import argparse

    configure_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    warm_up()
    print(f"[model={MODEL}]")
    result = ask(args.question)
    print(result.answer_text)
    for chart in result.charts:
        print()
        print(f"[chart: {chart.chart_type}] {chart.title}")
        print(f"  labels: {chart.labels}")
        print(f"  values: {chart.values}")
    if result.citations:
        print()
        print("[citations]")
        for c in result.citations:
            print(f"  {c.chunk_id} ({c.source}, page {c.page})")
    if result.tool_citations:
        print()
        print("[tool citations]")
        for tc in result.tool_citations:
            print(f"  {tc.description} (tool={tc.tool_name}, params={tc.parameters})")
    if result.downloads:
        print()
        print("[downloads]")
        for d in result.downloads:
            print(f"  {d.file_name} (status={d.status}, url={d.url})")


if __name__ == "__main__":
    main()

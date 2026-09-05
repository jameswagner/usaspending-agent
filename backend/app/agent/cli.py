"""CLI entry point.

Usage:
  python -m backend.app.agent --question "What is a prime award?"
"""
from __future__ import annotations

from .clients import MODEL
from .orchestrator import ask


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()

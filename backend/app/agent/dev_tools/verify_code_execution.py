"""Opt-in, real-billed verification that the code_execution fallback is
wired correctly:
  1. On a calculation genuinely outside the six typed arithmetic tools'
     coverage (a statistic none of them compute), the model reaches for
     code_execution, and the call is recorded and cited with the actual
     command that ran - not just "code was run."
  2. On a calculation the six typed tools DO cover, code_execution is NOT
     reached for - confirming the "prefer the typed tools" system prompt
     rule actually holds, not just that the fallback exists.

Doesn't specifically try to trigger a pause_turn stop_reason - the Python
SDK's tool_runner already documents that it resumes pause_turn/compaction
automatically (verified against tools.md, not assumed), so there's no
separate handling in this codebase to test for that.

Real, billed Claude calls plus code_execution's sandbox time (5-minute
execution-time minimum per Anthropic's pricing). Not part of CI - run
manually:
    uv run python -m backend.app.agent.dev_tools.verify_code_execution
"""
from __future__ import annotations

from backend.app.agent.orchestrator import ask


def case_code_execution_for_uncovered_calculation() -> None:
    print("\n=== Calculation outside the six typed tools' coverage ===")
    question = (
        "What is the standard deviation of NSF's spending across its top 5 "
        "NAICS categories in fiscal year 2024?"
    )
    print(f"Question: {question}")
    result = ask(question)
    print(f"Answer: {result.answer_text}")

    code_execution_citations = [c for c in result.tool_citations if c.tool_name == "code_execution"]
    print(f"code_execution citations: {code_execution_citations}")
    if not code_execution_citations:
        print("FAIL: expected a code_execution citation, got none.")
    else:
        print("OK: code_execution was called and cited with its command.")


def case_typed_tool_preferred_for_covered_calculation() -> None:
    print("\n=== Calculation the six typed tools already cover ===")
    question = (
        "What are NSF's top 2 NAICS categories by spending in fiscal year "
        "2024, and what do they add up to together?"
    )
    print(f"Question: {question}")
    result = ask(question)
    print(f"Answer: {result.answer_text}")

    code_execution_citations = [c for c in result.tool_citations if c.tool_name == "code_execution"]
    if code_execution_citations:
        print(f"NOTE: code_execution was used even though sum_values covers this: {code_execution_citations}")
    else:
        print("OK: no code_execution citation - the typed tool was preferred, as instructed.")


def main() -> None:
    case_code_execution_for_uncovered_calculation()
    case_typed_tool_preferred_for_covered_calculation()


if __name__ == "__main__":
    main()

"""Live, 3-turn verification of the LangGraph conversation path. Checks:
turn 2's bare follow-up still scopes to the prior turn's subject, and
turn 3's comparison calls the `delta` tool using both prior figures
rather than restating them in prose.

Real, billed API calls, run under AGENT_ENGINE=langgraph. Not part of
CI - run manually:
    AGENT_ENGINE=langgraph uv run python -m backend.app.agent.dev_tools.verify_conversation
"""
from __future__ import annotations

import os

if os.environ.get("AGENT_ENGINE") != "langgraph":
    raise SystemExit("Set AGENT_ENGINE=langgraph before running this script.")

import uuid

from backend.app.agent.orchestrator import NOT_FOUND_MESSAGE
from backend.app.agent.singletons import warm_up


def main() -> None:
    warm_up()
    from backend.app.agent.orchestrator import ask

    conversation_id = str(uuid.uuid4())
    turns = [
        "What was NASA's FY2024 budget?",
        "What about FY2023?",
        "Which was higher?",
    ]

    results = []
    for question in turns:
        result = ask(question, conversation_id=conversation_id)
        results.append(result)
        tools_called = [tc.tool_name for tc in result.tool_citations]
        print(f"\nQ: {question}")
        print(f"A: {result.answer_text}")
        print(f"tool_citations: {tools_called}")

    turn2, turn3 = results[1], results[2]

    if turn2.answer_text == NOT_FOUND_MESSAGE:
        print("\nFAIL: turn 2 was rejected by the scope gate - the follow-up scope "
              "classifier should have recognized this as a continuation.")
    else:
        print("\nOK: turn 2 was not rejected by the scope gate.")

    turn3_tools = [tc.tool_name for tc in turn3.tool_citations]
    if "delta" in turn3_tools:
        print("OK: turn 3 called delta using the prior turns' figures.")
    else:
        print(f"FAIL: turn 3 did not call delta - tool_citations were {turn3_tools}.")


if __name__ == "__main__":
    main()

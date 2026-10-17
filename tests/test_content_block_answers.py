"""Models with thinking (Haiku 5.5+) return the final answer as a list of content blocks."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.orchestrator import _ask_langgraph
from backend.app.agent.scope import _render_recent_exchanges

BLOCK_ANSWER = AIMessage(content=[
    {"type": "thinking", "thinking": "Work it out.", "signature": "sig"},
    {"type": "text", "text": "NSF's FY2024 budget was $9.9B."},
])


def test_ask_returns_the_text_block_of_a_block_list_answer():
    graph = MagicMock()
    graph.get_state.return_value.values = {"messages": []}
    graph.invoke.return_value = {"messages": [BLOCK_ANSWER]}
    with patch("backend.app.agent.orchestrator._get_conversation_graph", return_value=graph), patch(
        "backend.app.agent.orchestrator._is_in_scope", return_value=True
    ):
        result = _ask_langgraph("What was NSF's FY2024 budget?", "conversation")
    assert result.answer_text == "NSF's FY2024 budget was $9.9B."


def test_scope_history_keeps_block_list_answers_and_skips_tool_calls():
    tool_call = AIMessage(
        content=[{"type": "text", "text": "Let me look."}],
        tool_calls=[{"name": "lookup_agency", "args": {"name": "NSF"}, "id": "1"}],
    )
    rendered = _render_recent_exchanges([HumanMessage(content="NSF budget?"), tool_call, BLOCK_ANSWER])
    assert rendered == "Prior turn: Q: NSF budget?\nA: NSF's FY2024 budget was $9.9B."

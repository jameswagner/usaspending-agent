from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage

from backend.app.agent import orchestrator


class _ThinkingOnlyGraph:
    """The final turn spent its whole output budget thinking (stop_reason max_tokens), so it has no text."""

    def get_state(self, config):
        return SimpleNamespace(values={})

    def invoke(self, state, config):
        return {"messages": [AIMessage(content=[{"type": "thinking", "thinking": "", "signature": "x"}])]}


def test_a_thinking_only_final_turn_returns_a_message_not_a_blank(monkeypatch):
    monkeypatch.setattr(orchestrator, "_get_conversation_graph", lambda toolset=None: _ThinkingOnlyGraph())
    monkeypatch.setattr(orchestrator, "_is_in_scope", lambda question, recent: True)

    result = orchestrator.ask("Which FY2025 contracts funded by NASA are largest?")

    assert result.answer_text == orchestrator.NO_ANSWER_MESSAGE

"""Fast, CI-gated coverage for the LangGraph conversation persistence
layer - no real model/API calls: a tmp_path SqliteSaver and a scripted
FakeMessagesListChatModel instead.
"""
from __future__ import annotations

import sqlite3

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import create_react_agent


def _make_saver(tmp_path) -> SqliteSaver:
    conn = sqlite3.connect(str(tmp_path / "test_conversations.db"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


class TestCheckpointerIsolation:
    def test_checkpoints_are_isolated_per_thread(self, tmp_path):
        saver = _make_saver(tmp_path)
        config_a = {"configurable": {"thread_id": "thread-a", "checkpoint_ns": ""}}
        config_b = {"configurable": {"thread_id": "thread-b", "checkpoint_ns": ""}}

        checkpoint_a = empty_checkpoint()
        checkpoint_a["channel_values"] = {"messages": ["A's message"]}
        saver.put(config_a, checkpoint_a, {"source": "input", "step": 0, "parents": {}}, {})

        checkpoint_b = empty_checkpoint()
        checkpoint_b["channel_values"] = {"messages": ["B's message"]}
        saver.put(config_b, checkpoint_b, {"source": "input", "step": 0, "parents": {}}, {})

        assert saver.get_tuple(config_a).checkpoint["channel_values"]["messages"] == ["A's message"]
        assert saver.get_tuple(config_b).checkpoint["channel_values"]["messages"] == ["B's message"]

    def test_unknown_thread_returns_none(self, tmp_path):
        saver = _make_saver(tmp_path)
        config = {"configurable": {"thread_id": "never-used", "checkpoint_ns": ""}}
        assert saver.get_tuple(config) is None


class TestReactAgentRoundTrip:
    def _make_graph(self, tmp_path, responses):
        saver = _make_saver(tmp_path)
        model = FakeMessagesListChatModel(responses=responses)
        return create_react_agent(model=model, tools=[], checkpointer=saver)

    def test_same_thread_reuses_history(self, tmp_path):
        graph = self._make_graph(tmp_path, [AIMessage(content="first answer"), AIMessage(content="second answer")])
        config = {"configurable": {"thread_id": "t1"}}

        graph.invoke({"messages": [{"role": "user", "content": "question 1"}]}, config=config)
        result = graph.invoke({"messages": [{"role": "user", "content": "question 2"}]}, config=config)

        contents = [m.content for m in result["messages"]]
        assert "question 1" in contents
        assert "question 2" in contents

    def test_different_thread_has_no_leakage(self, tmp_path):
        graph = self._make_graph(tmp_path, [AIMessage(content="first answer"), AIMessage(content="second answer")])
        graph.invoke(
            {"messages": [{"role": "user", "content": "question 1"}]},
            config={"configurable": {"thread_id": "t1"}},
        )

        result = graph.invoke(
            {"messages": [{"role": "user", "content": "question 3"}]},
            config={"configurable": {"thread_id": "t2"}},
        )

        contents = [m.content for m in result["messages"]]
        assert "question 1" not in contents

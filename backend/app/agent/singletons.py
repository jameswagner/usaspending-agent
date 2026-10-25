"""Lazily-constructed singleton clients shared across the agent package, and
the constants that configure them.
"""
from __future__ import annotations

import os
import sqlite3
import types

import anthropic
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt, wrap_model_call
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langsmith.wrappers import wrap_anthropic

from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.usaspending import USASpendingClient

load_dotenv()

ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID")

# Swappable via env var so models can be compared on the same test questions.
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-5-5")

# "current" (LANGGRAPH_TOOLS) or "consolidated" (CONSOLIDATED_TOOLS); both graphs are built at warm_up.
AGENT_TOOLSET = os.environ.get("AGENT_TOOLSET", "consolidated")

# Same floor as main.py, calibrated via dev_tools/calibrate_threshold.py
# (data-driven optimum -1.89, using -2.0 for a small safety margin).
RERANK_CONFIDENCE_THRESHOLD = -2.0

# Shared by both cache_control breakpoints below (system prompt, tools).
CACHE_TTL = "1h"

NAICS_CHROMA_DB_DIR = os.environ.get("NAICS_CHROMA_DB_DIR", "./data/chroma_naics")
NAICS_WHOOSH_INDEX_DIR = os.environ.get("NAICS_WHOOSH_INDEX_DIR", "./data/whoosh_naics")
PSC_CHROMA_DB_DIR = os.environ.get("PSC_CHROMA_DB_DIR", "./data/chroma_psc")
PSC_WHOOSH_INDEX_DIR = os.environ.get("PSC_WHOOSH_INDEX_DIR", "./data/whoosh_psc")
CFDA_CHROMA_DB_DIR = os.environ.get("CFDA_CHROMA_DB_DIR", "./data/chroma_cfda")
CFDA_WHOOSH_INDEX_DIR = os.environ.get("CFDA_WHOOSH_INDEX_DIR", "./data/whoosh_cfda")

CONVERSATIONS_DB_PATH = os.environ.get("CONVERSATIONS_DB_PATH", "./data/conversations.db")

# A hard cutoff, not summarization - deliberately simple for v1.
CONVERSATION_HISTORY_TURNS = int(os.environ.get("CONVERSATION_HISTORY_TURNS", "10"))

_client: anthropic.Anthropic | None = None
_retriever: HybridRetriever | None = None
_naics_retriever: HybridRetriever | None = None
_psc_retriever: HybridRetriever | None = None
_cfda_retriever: HybridRetriever | None = None
_usaspending_client: USASpendingClient | None = None

# Unlike every other singleton in this file, these three are never lazily
# constructed on first access (if x is None: x = Construct(), no lock) -
# that exact pattern is the root cause of #44/#45/#54, all real concurrency
# bugs. warm_up() assigns these exactly once, at startup, before any
# request is served, so there's no request-time race to have.
_checkpointer: SqliteSaver | None = None
_chat_model: ChatAnthropic | None = None
_conversation_graphs: dict[str, object] = {}


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
        # wrap_anthropic patches client.messages.create/.stream and
        # client.beta.messages.create/.parse in place - it does NOT wrap
        # tool_runner directly (no such method exists to patch), but
        # BetaToolRunner's non-streaming path calls the now-patched
        # self._client.beta.messages.parse(...) internally (verified
        # against the installed SDK's source, since the docs only show
        # bare messages.create usage), so tool_runner calls still get
        # fully traced - every tool_use/tool_result block, not just the
        # question and final answer @traceable(agent_ask) alone captured
        # before this. Only meaningful for the non-streaming path this
        # app actually uses (tool_runner is never called with stream=True
        # here) - client.beta.messages.stream isn't wrapped at all.
        raw_client = anthropic.Anthropic(default_headers=headers)
        # wrap_anthropic (langsmith 0.12.1/0.12.2, confirmed both) always
        # tries to patch client.completions.create with no hasattr guard -
        # unlike its beta.messages patches, which do guard - but this SDK
        # version has no `completions` attribute at all (the legacy Text
        # Completions API is gone), so it raises AttributeError outright.
        # A real upstream bug, not something to route around by catching
        # the exception: give it a placeholder to patch instead, which is
        # never actually called since nothing here uses the Completions API.
        if not hasattr(raw_client, "completions"):
            raw_client.completions = types.SimpleNamespace(create=lambda *a, **k: None)
        _client = wrap_anthropic(raw_client)
    return _client


def _get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever


def _get_naics_retriever() -> HybridRetriever:
    global _naics_retriever
    if _naics_retriever is None:
        _naics_retriever = HybridRetriever(
            chroma_db_dir=NAICS_CHROMA_DB_DIR, whoosh_index_dir=NAICS_WHOOSH_INDEX_DIR
        )
    return _naics_retriever


def _get_psc_retriever() -> HybridRetriever:
    global _psc_retriever
    if _psc_retriever is None:
        _psc_retriever = HybridRetriever(chroma_db_dir=PSC_CHROMA_DB_DIR, whoosh_index_dir=PSC_WHOOSH_INDEX_DIR)
    return _psc_retriever


def _get_cfda_retriever() -> HybridRetriever:
    global _cfda_retriever
    if _cfda_retriever is None:
        _cfda_retriever = HybridRetriever(chroma_db_dir=CFDA_CHROMA_DB_DIR, whoosh_index_dir=CFDA_WHOOSH_INDEX_DIR)
    return _cfda_retriever


def _get_usaspending_client() -> USASpendingClient:
    global _usaspending_client
    if _usaspending_client is None:
        _usaspending_client = USASpendingClient()
    return _usaspending_client


def _count_human_turns(messages: list[BaseMessage]) -> int:
    """trim_messages calls this once on the whole candidate list, not per-message."""
    return sum(1 for m in messages if isinstance(m, HumanMessage))


@wrap_model_call
def _trim_history(request: ModelRequest, handler):
    """Trims only the model's input; editing state messages would overwrite the checkpointer's persisted history."""
    trimmed = trim_messages(
        request.messages,
        max_tokens=CONVERSATION_HISTORY_TURNS,
        token_counter=_count_human_turns,
        strategy="last",
        start_on="human",
    )
    return handler(request.override(messages=trimmed))


def _get_conversation_graph(toolset: str | None = None):
    assert _conversation_graphs, "warm_up() must run before _get_conversation_graph()"
    return _conversation_graphs[toolset or AGENT_TOOLSET]


def _get_checkpointer() -> SqliteSaver:
    assert _checkpointer is not None, "warm_up() must run before _get_checkpointer()"
    return _checkpointer


def warm_up() -> None:
    """Pre-load the retriever's models and both clients once, at server
    startup, instead of paying that cost on whichever request happens to
    be first.
    """
    _get_retriever()
    _get_naics_retriever()
    _get_psc_retriever()
    _get_cfda_retriever()
    _get_usaspending_client()
    _get_client()

    # Deferred imports: both orchestrator and langgraph_tools (via
    # tools/*) import from this module, so importing either at module
    # load time would be circular. By call time (warm_up() only runs
    # from main.py's lifespan(), after every module has finished
    # importing) the cycle doesn't exist.
    from .consolidated_tools import CONSOLIDATED_TOOLS
    from .langgraph_tools import LANGGRAPH_TOOLS
    from .orchestrator import _build_consolidated_system_prompt, _build_system_prompt

    global _checkpointer, _chat_model
    conn = sqlite3.connect(CONVERSATIONS_DB_PATH, check_same_thread=False)
    _checkpointer = SqliteSaver(conn)
    headers = {"anthropic-workspace-id": ANTHROPIC_WORKSPACE_ID} if ANTHROPIC_WORKSPACE_ID else None
    _chat_model = ChatAnthropic(model=MODEL, max_tokens=2048, default_headers=headers)

    _conversation_graphs["current"] = _build_conversation_graph(LANGGRAPH_TOOLS, _build_system_prompt)
    _conversation_graphs["consolidated"] = _build_conversation_graph(CONSOLIDATED_TOOLS, _build_consolidated_system_prompt)


def _build_conversation_graph(tool_list: list, build_prompt):
    cache_control = {"type": "ephemeral", "ttl": CACHE_TTL}
    # One breakpoint on the last tool caches the whole tools array.
    last_tool = tool_list[-1]
    tools = [*tool_list[:-1], last_tool.model_copy(update={"extras": {**(last_tool.extras or {}), "cache_control": cache_control}})]

    @dynamic_prompt
    def _system_prompt(request: ModelRequest) -> SystemMessage:
        # Rebuilt per call for date grounding; caching needs a content block, not a bare string.
        return SystemMessage(content=[{"type": "text", "text": build_prompt(), "cache_control": cache_control}])

    return create_agent(
        model=_chat_model,
        tools=tools,
        middleware=[_system_prompt, _trim_history],
        checkpointer=_checkpointer,
    )

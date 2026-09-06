"""Lazily-constructed singleton clients shared across the agent package, and
the constants that configure them.
"""
from __future__ import annotations

import os
import types

import anthropic
from dotenv import load_dotenv
from langsmith.wrappers import wrap_anthropic

from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.usaspending_client import USASpendingClient

load_dotenv()

ANTHROPIC_WORKSPACE_ID = os.environ.get("ANTHROPIC_WORKSPACE_ID")

# Swappable via env var so Haiku 4.5 vs Sonnet 5 can be compared on the same
# test questions before picking one for tool-calling specifically.
MODEL = os.environ.get("AGENT_MODEL", "claude-haiku-4-5")

# Same floor as main.py, calibrated via dev_tools/calibrate_threshold.py
# (data-driven optimum -1.89, using -2.0 for a small safety margin).
RERANK_CONFIDENCE_THRESHOLD = -2.0

_client: anthropic.Anthropic | None = None
_retriever: HybridRetriever | None = None
_usaspending_client: USASpendingClient | None = None


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


def _get_usaspending_client() -> USASpendingClient:
    global _usaspending_client
    if _usaspending_client is None:
        _usaspending_client = USASpendingClient()
    return _usaspending_client


def warm_up() -> None:
    """Pre-load the retriever's models and both clients once, at server
    startup, instead of paying that cost on whichever request happens to
    be first.
    """
    _get_retriever()
    _get_usaspending_client()
    _get_client()

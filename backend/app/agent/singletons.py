"""Lazily-constructed singleton clients shared across the agent package, and
the constants that configure them.
"""
from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

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
        _client = anthropic.Anthropic(default_headers=headers)
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

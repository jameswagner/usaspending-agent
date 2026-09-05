"""Tool-calling agent over USASpending question-answering.

Five tools: search_guide (wraps HybridRetriever, for conceptual/definitional
questions), lookup_agency, get_spending_by_category, get_spending_over_time,
and search_awards (all four live-data tools backed by usaspending_client).

Usage:
  python -m backend.app.agent --question "What is a prime award?"

Submodules:
  clients.py           - lazily-constructed singleton clients + shared config
  tools.py              - the five @beta_tool functions and their _raw variants
  response_shaping.py  - pure post-processing: charts, citations, fiscal-year math
  scope.py              - the in-scope pre-filter gate
  orchestrator.py       - the system prompt, AgentResult, and ask()
  cli.py                - the --question CLI entry point

ask() and AgentResult are re-exported here since they're what nearly every
external caller (main.py, tests, dev_tools scripts) needs.
"""
from __future__ import annotations

from .orchestrator import AgentResult, ask

__all__ = ["AgentResult", "ask"]

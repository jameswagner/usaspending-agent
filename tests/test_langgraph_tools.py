"""Schema-sanity coverage for langgraph_tools.py - catches a
tool(parse_docstring=True) mismatch between a docstring's Args: block
and its function's real signature, across all wrapped tools at once.
"""
from __future__ import annotations

import inspect

from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS


def test_langgraph_tools_is_nonempty():
    assert len(LANGGRAPH_TOOLS) == 25


def test_every_tool_schema_matches_its_own_signature():
    mismatches = []
    for wrapped in LANGGRAPH_TOOLS:
        expected = set(inspect.signature(wrapped.func).parameters)
        actual = set(wrapped.args_schema.model_fields)
        if expected != actual:
            mismatches.append((wrapped.name, expected, actual))

    assert not mismatches, "\n".join(
        f"{name}: signature has {expected}, schema has {actual}" for name, expected, actual in mismatches
    )

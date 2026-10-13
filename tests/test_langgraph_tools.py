"""Schema-sanity coverage for langgraph_tools.py - catches a
tool(parse_docstring=True) mismatch between a docstring's Args: block
and its function's real signature, across all wrapped tools at once.
"""
from __future__ import annotations

import inspect

from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS


def test_langgraph_tools_is_nonempty():
    assert len(LANGGRAPH_TOOLS) == 29


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


def test_unknown_psc_list_cannot_silently_widen_query_spending():
    wrapped = next(tool for tool in LANGGRAPH_TOOLS if tool.name == "query_spending")
    result = wrapped.invoke({
        "level": "award", "group_by": "time", "start_year": 2024, "end_year": 2024,
        "psc_codes": ["DA01", "D302"],
    })
    assert result == "This query failed: unsupported tool argument(s): psc_codes. Check the tool schema."


def test_unknown_download_filter_cannot_silently_widen_csv():
    wrapped = next(tool for tool in LANGGRAPH_TOOLS if tool.name == "download_records")
    result = wrapped.invoke({
        "start_fiscal_year": 2024, "end_fiscal_year": 2024,
        "psc_codess": ["DA01", "D302"],
    })
    assert result == "This download failed: unsupported tool argument(s): psc_codess. Check the tool schema."

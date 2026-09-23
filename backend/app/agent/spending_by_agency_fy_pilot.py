"""Deterministic pipeline for "how much did [agency] spend [optionally: by
object class] in FY[year]?" (issue #233, child of #228's "move work out of
the tool-calling loop" epic). Intercepts before the tool loop starts - see
_ask_langgraph/_run_graph_stream, which check
_looks_like_agency_fy_spending_request(question) right after the existing
_is_in_scope gate - rather than relying on the model to pick
get_spending_explorer_breakdown correctly and only ever call it once.

Reuses get_spending_explorer_breakdown_raw (tools/spending_explorer.py) for
the actual API call/agency resolution/validation, and _record_tool_call +
_build_result (orchestrator.py) for chart/citation building, so this pilot's
output goes through the exact same should_chart/build_tool_citation branches
a real tool call would.

Any USASpendingAPIError from the API call (unresolved agency, an
in-progress fiscal quarter with no data yet, etc.) falls through to the
normal tool loop rather than surfacing an error - the model there can pick a
different quarter or explain the gap itself.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Literal

from pydantic import BaseModel

from backend.app.usaspending import USASpendingAPIError

from .response_shaping import current_fiscal_year, year_label
from .singletons import MODEL, _get_client
from .tools import _record_tool_call, _tool_call_log
from .tools.spending_explorer import (
    _format_spending_explorer_results,
    get_spending_explorer_breakdown_raw,
)

logger = logging.getLogger(__name__)

_SPEND_VERB_PATTERN = re.compile(r"\b(spend|spent|spending|obligat\w*)\b", re.IGNORECASE)
_YEAR_TOKEN_PATTERN = re.compile(r"\bfy ?\d{4}\b|\b(19|20)\d{2}\b", re.IGNORECASE)


def _looks_like_agency_fy_spending_request(question: str) -> bool:
    return bool(_SPEND_VERB_PATTERN.search(question) and _YEAR_TOKEN_PATTERN.search(question))


def pilot_enabled() -> bool:
    """Read at call time so the eval script can toggle it per-call; defaults off pending #245 instrumentation."""
    return os.environ.get("SPENDING_BY_AGENCY_FY_PILOT_ENABLED", "false").lower() == "true"


class AgencyFYSpendingIntent(BaseModel):
    other_filters_present: bool = True
    agency_raw: str | None = None
    fiscal_year: int | None = None
    group_by: Literal["object_class"] | None = None


_EXTRACT_TOOL_NAME = "extract_agency_fy_spending_intent"
_EXTRACT_TOOL = {
    "name": _EXTRACT_TOOL_NAME,
    "description": (
        "Extract the agency and single fiscal year for a 'how much did [agency] spend in "
        "FY[year]?' style question, and whether it asks for an object-class breakdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "other_filters_present": {
                "type": "boolean",
                "description": (
                    "True if the question mentions more than one agency (including any "
                    "comparison between agencies), more than one fiscal year, or any filter "
                    "beyond a single agency + single fiscal year - a recipient, NAICS/PSC code, "
                    "location, award type, a dollar-amount threshold, or a date range narrower "
                    "than a full fiscal year. False only for a clean single-agency, "
                    "single-fiscal-year question with no other filter."
                ),
            },
            "agency_raw": {
                "type": "string",
                "description": "Agency name/abbreviation as stated. Required even if other_filters_present is true.",
            },
            "fiscal_year": {
                "type": "integer",
                "description": "The fiscal year asked about (or the first one, if other_filters_present is true).",
            },
            "group_by": {
                "type": "string",
                "enum": ["object_class"],
                "description": "Only set if the question explicitly asks for a breakdown by object class.",
            },
        },
        "required": ["other_filters_present", "agency_raw", "fiscal_year"],
    },
}


def _extract_agency_fy_spending_intent(question: str) -> AgencyFYSpendingIntent | None:
    """One forced-tool-choice call, same pattern as download_pilot.py's
    _extract_download_intent. Returns None on anything that isn't a clean,
    single-agency/single-year parse - falling through to the tool loop is
    always safer than guessing."""
    system = (
        f"Today's fiscal year is FY{current_fiscal_year()}. Use it for relative phrases like "
        "'this year' or 'most recent year' if the question doesn't state one explicitly."
    )
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": _EXTRACT_TOOL_NAME},
            messages=[{"role": "user", "content": question}],
        )
    except Exception:
        logger.exception("Agency/FY spending intent extraction call failed for question: %r", question)
        return None

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        return None
    try:
        intent = AgencyFYSpendingIntent.model_validate(tool_use.input)
    except Exception:
        logger.exception("Agency/FY spending intent extraction returned unparseable input: %r", tool_use.input)
        return None
    if intent.other_filters_present or intent.agency_raw is None or intent.fiscal_year is None:
        return None
    return intent


def _synthesize_answer(question: str, agency_raw: str, fiscal_year: int, response) -> str:
    """One plain (non-tool-use) synthesis call turning the structured
    SpendingExplorerResponse into the markdown answer - reuses the same
    text formatting the real get_spending_explorer_breakdown tool shows the
    model, so this call is doing phrasing, not arithmetic."""
    data_text = _format_spending_explorer_results(response)
    system = (
        "You answer a single US government spending question from already-fetched data. "
        "Be concise - a sentence or two, in markdown, with dollar amounts formatted like "
        "$1,234,567. Don't mention tools, APIs, or how the data was fetched."
    )
    user_content = (
        f"Question: {question}\n\n"
        f"Data (spending explorer, {agency_raw}, {year_label('fiscal', fiscal_year)}):\n{data_text}"
    )
    llm_response = _get_client().messages.create(
        model=MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": user_content}],
    )
    text_block = next((b for b in llm_response.content if b.type == "text"), None)
    return text_block.text if text_block is not None else data_text


def handle_agency_fy_spending_request(question: str, conversation_id: str):
    """Returns an AgentResult, or None to signal "not a confident enough
    match - fall through to the normal tool-calling loop unchanged" (no
    single clean agency/year, unresolved agency, or an API error such as an
    in-progress fiscal quarter with no data yet). Local import of
    AgentResult/_build_result avoids a circular import with orchestrator.py,
    which imports this module."""
    from .orchestrator import _build_result

    intent = _extract_agency_fy_spending_intent(question)
    if intent is None:
        return None

    group_by = intent.group_by or "agency"
    try:
        response = get_spending_explorer_breakdown_raw(
            group_by, intent.fiscal_year, "4", agency=intent.agency_raw
        )
        context = {
            "group_by": group_by,
            "fiscal_year": intent.fiscal_year,
            "quarter": "4",
            "agency": intent.agency_raw,
        }
    except USASpendingAPIError as e:
        logger.info("Agency/FY spending pilot falling through to tool loop for %r: %s", question, e)
        return None

    if not response.results:
        return None

    answer_text = _synthesize_answer(question, intent.agency_raw, intent.fiscal_year, response)

    _tool_call_log.set([])
    _record_tool_call("get_spending_explorer_breakdown", response, context)
    return _build_result(answer_text, conversation_id)

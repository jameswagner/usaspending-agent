"""Deterministic download-request pipeline (issue #238, child of #228's
"move work out of the tool-calling loop" epic). Intercepts a "download
this as a CSV" question before the tool loop starts - see
_ask_langgraph/_run_graph_stream, which both check
_looks_like_download_request(question) right after their existing
_is_in_scope gate - rather than adding another @beta_tool the model has
to be prompted into choosing correctly.

Narrowly scoped to POST /api/v2/download/awards/ only, per #238. Every
other download endpoint (transactions, accounts, contract, IDV, disaster,
search, bulk_download) gets a fixed "not yet supported" answer instead of
best-effort handling - see _UNSUPPORTED_DOWNLOAD_PATTERN.
"""
from __future__ import annotations

import logging
import time
from typing import Literal

from pydantic import BaseModel, ValidationError

from backend.app.usaspending import USASpendingAPIError, USASpendingClient

from .response_shaping import DownloadSpec, current_fiscal_year
from .singletons import MODEL, _get_client, _get_usaspending_client
from .tool_filters import _build_filters

logger = logging.getLogger(__name__)

_DOWNLOAD_INTENT_PATTERN = (
    "download", "export", "csv", "spreadsheet", "give me the data", "give me a file",
    "raw data", "data file",
)

# Endpoints #238 explicitly defers - matched before intent extraction runs,
# so an unsupported request never reaches (and never burns) the extraction
# call. Deliberately keyword-based, same "cheap, no LLM call" reasoning as
# _looks_like_download_request below.
_UNSUPPORTED_DOWNLOAD_PATTERN = {
    "transaction": "transaction-level data",
    "account": "account-level data",
    "sub-award": "sub-award data",
    "subaward": "sub-award data",
    "idv": "IDV data",
    "indefinite delivery": "IDV data",
    "disaster": "disaster/relief-specific data",
}

_POLL_INTERVAL_SECONDS = 2
_POLL_TIMEOUT_SECONDS = 45

# Small, fixed set - #238 explicitly rules out free-form column selection
# for this pilot.
_DOWNLOAD_COLUMNS = [
    "award_id_piid",
    "award_id_fain",
    "recipient_name",
    "total_obligated_amount",
    "period_of_performance_start_date",
    "awarding_agency_name",
]


def _looks_like_download_request(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _DOWNLOAD_INTENT_PATTERN)


def _unsupported_download_label(question: str) -> str | None:
    q = question.lower()
    for term, label in _UNSUPPORTED_DOWNLOAD_PATTERN.items():
        if term in q:
            return label
    return None


class DownloadIntent(BaseModel):
    agency_raw: str | None = None
    time_period_type: Literal["fiscal", "calendar"] = "fiscal"
    start_year: int | None = None
    end_year: int | None = None
    award_type: str | None = None


_EXTRACT_TOOL_NAME = "extract_download_intent"
_EXTRACT_TOOL = {
    "name": _EXTRACT_TOOL_NAME,
    "description": (
        "Extract the agency and year range (fiscal or calendar) the user wants a spending "
        "award CSV download for. If the question doesn't name a specific agency or a resolvable "
        "year range, omit those fields rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "agency_raw": {"type": "string", "description": "Agency name/abbreviation as stated, or omitted if none is named."},
            "time_period_type": {"type": "string", "enum": ["fiscal", "calendar"]},
            "start_year": {"type": "integer", "description": "First year of the range, or omitted if unclear."},
            "end_year": {"type": "integer", "description": "Last year of the range, or omitted if unclear."},
            "award_type": {
                "type": "string",
                "enum": ["contracts", "grants", "loans"],
                "description": "Omit unless the question specifically names one award type.",
            },
        },
    },
}


def _extract_download_intent(question: str) -> DownloadIntent | None:
    """One forced-tool-choice call - no structured-output helper exists
    elsewhere in this codebase yet (scope.py only ever parses a bare
    YES/NO), so this is the first. Returns None on anything that isn't a
    clean parse - a download pipeline that guesses a wrong agency or year
    range is worse than falling through to the normal tool loop, where the
    model can ask a clarifying question instead."""
    system = (
        f"Today's fiscal year is FY{current_fiscal_year()}. Use it for relative phrases "
        "like 'this year' or 'most recent year' if the question doesn't state one explicitly."
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
        logger.exception("Download intent extraction call failed for question: %r", question)
        return None

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        return None
    try:
        intent = DownloadIntent.model_validate(tool_use.input)
    except ValidationError:
        logger.warning("Download intent extraction returned unparseable input: %r", tool_use.input)
        return None
    if intent.start_year is None or intent.end_year is None:
        return None
    return intent


def _poll_until_finished(client: USASpendingClient, file_name: str):
    """Blocks up to _POLL_TIMEOUT_SECONDS so the common small-filter case
    (confirmed live: ~4s for a one-agency, one-month pull) returns the
    finished file_url within a single turn. Status moves ready -> running ->
    finished/failed (api_contracts/contracts/v2/download/status.md) - both
    ready and running mean "keep polling," confirmed live: a freshly queued
    job answers "ready" for the first several seconds, and a poll loop that
    only recognized "running" as in-progress stopped after a single check.
    Returns the last status seen - caller decides what a non-"finished"
    result (still ready/running, or failed) means for the answer text."""
    elapsed = 0.0
    status = client.get_download_status(file_name)
    while status.status in ("ready", "running") and elapsed < _POLL_TIMEOUT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS
        status = client.get_download_status(file_name)
    return status


def handle_download_request(question: str, conversation_id: str):
    """Returns an AgentResult, or None to signal "not a confident enough
    download request - fall through to the normal tool-calling loop
    unchanged" (ambiguous year range, agency that doesn't resolve). Local
    import of AgentResult avoids a circular import with orchestrator.py,
    which imports this module."""
    from .orchestrator import AgentResult

    unsupported = _unsupported_download_label(question)
    if unsupported is not None:
        return AgentResult(
            answer_text=(
                f"Downloading {unsupported} isn't supported yet — only award-level CSV "
                "exports are, for now."
            ),
            conversation_id=conversation_id,
        )

    intent = _extract_download_intent(question)
    if intent is None:
        return None

    client = _get_usaspending_client()
    agency_name = None
    if intent.agency_raw:
        match = client.find_agency_by_name(intent.agency_raw)
        if match is None:
            return None
        agency_name = match.agency_name

    try:
        filters = _build_filters(
            client,
            agency_name,
            intent.time_period_type,
            intent.start_year,
            intent.end_year,
            award_type=intent.award_type,
            scope_required=False,
        )
        job = client.download_awards(filters, _DOWNLOAD_COLUMNS)
        status = _poll_until_finished(client, job.file_name)
    except USASpendingAPIError as e:
        logger.warning("Download pipeline failed for question %r: %s", question, e)
        return AgentResult(answer_text=f"This download failed: {e}.", conversation_id=conversation_id)

    if status.status == "finished":
        answer_text = (
            f"Your download is ready: {status.total_rows or 0} rows in {status.file_name}.\n{status.file_url}"
        )
        download = DownloadSpec(
            file_name=status.file_name, url=status.file_url, status=status.status, total_rows=status.total_rows
        )
    elif status.status == "failed":
        answer_text = f"This download failed to generate: {status.message or 'no further detail from the API.'}"
        download = DownloadSpec(file_name=status.file_name, url=None, status=status.status, total_rows=status.total_rows)
    else:
        answer_text = (
            f"Your download is still generating ({status.file_name}). Check back shortly at "
            f"{status.file_url}, or poll the status yourself: {job.status_url}"
        )
        download = DownloadSpec(file_name=status.file_name, url=None, status=status.status, total_rows=status.total_rows)

    return AgentResult(answer_text=answer_text, conversation_id=conversation_id, downloads=[download])

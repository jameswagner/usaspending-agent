"""Deterministic pre-tool-loop download-request pipeline, scoped to POST /api/v2/download/awards/ only."""
from __future__ import annotations

import json
import logging
import time
from typing import Literal

from langsmith.run_helpers import get_current_run_tree
from pydantic import BaseModel, ValidationError

from backend.app.usaspending import (
    BASE_URL,
    AdvancedFilters,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

from .response_shaping import (
    DownloadSpec,
    ToolCitation,
    current_fiscal_year,
    year_label,
)
from .scope import _render_recent_exchanges
from .singletons import MODEL, _get_client, _get_usaspending_client
from .tool_filters import _build_filters

logger = logging.getLogger(__name__)

_DOWNLOAD_INTENT_PATTERN = (
    "download", "export", "csv", "spreadsheet", "give me the data", "give me a file",
    "raw data", "data file",
)

# Endpoints this pilot defers - matched before the extraction call runs, so an unsupported request never burns one.
_UNSUPPORTED_DOWNLOAD_PATTERN = {
    "transaction": "transaction-level data",
    "account": "account-level data",
    "sub-award": "sub-award data",
    "subaward": "sub-award data",
    "idv": "IDV data",
    "indefinite delivery": "IDV data",
    "disaster": "disaster/relief-specific data",
}

_POLL_INTERVAL_SECONDS = 4
_POLL_TIMEOUT_SECONDS = 90

# Fixed - this pilot rules out free-form column selection.
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


# Prefixes handle_download_request's own answer_text always starts with - lets the caller recognize a download follow-up.
_DOWNLOAD_ANSWER_PREFIXES = ("Your download is", "This download failed", "Downloading ")


def _is_download_followup(recent_messages: list | None) -> bool:
    for message in reversed(recent_messages or []):
        if getattr(message, "type", None) == "ai" and isinstance(message.content, str):
            return message.content.startswith(_DOWNLOAD_ANSWER_PREFIXES)
    return False


def _unsupported_download_label(question: str) -> str | None:
    q = question.lower()
    for term, label in _UNSUPPORTED_DOWNLOAD_PATTERN.items():
        if term in q:
            return label
    return None


class DownloadIntent(BaseModel):
    # Required in the tool schema below - forces the model to actively decide, not default to True.
    wants_download: bool = True
    agency_raw: str | None = None
    time_period_type: Literal["fiscal", "calendar"] = "fiscal"
    start_year: int | None = None
    end_year: int | None = None
    # Set together instead of start_year/end_year for a period narrower than a full year.
    start_date: str | None = None
    end_date: str | None = None
    award_type: str | None = None


_EXTRACT_TOOL_NAME = "extract_download_intent"
_EXTRACT_TOOL = {
    "name": _EXTRACT_TOOL_NAME,
    "description": (
        "Extract the agency and time period the user wants a spending award CSV download for. "
        "If the question doesn't name a specific agency or a resolvable time period, omit those "
        "fields rather than guessing."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "wants_download": {
                "type": "boolean",
                "description": "True only if this specific question wants an actual exported file, not "
                "just spending information - e.g. a breakdown, comparison, or 'how much' question is "
                "false even if it follows a download in the conversation and even if it names an agency/year.",
            },
            "agency_raw": {"type": "string", "description": "Agency name/abbreviation as stated, or omitted if none is named."},
            "time_period_type": {"type": "string", "enum": ["fiscal", "calendar"]},
            "start_year": {
                "type": "integer",
                "description": "First year of the range. Only use this (with end_year) when the "
                "question asks for a WHOLE fiscal or calendar year - omit it and use start_date/"
                "end_date instead whenever the question names anything narrower (a specific month, "
                "quarter, or date range).",
            },
            "end_year": {"type": "integer", "description": "Last year of the range, paired with start_year."},
            "start_date": {
                "type": "string",
                "description": "YYYY-MM-DD. Use this (with end_date) instead of start_year/end_year "
                "whenever the question names a period narrower than a full year - e.g. 'January "
                "2024' -> start_date='2024-01-01', end_date='2024-01-31'.",
            },
            "end_date": {"type": "string", "description": "YYYY-MM-DD, paired with start_date."},
            "award_type": {
                "type": "string",
                "enum": ["contracts", "grants", "loans"],
                "description": "Omit unless the question specifically names one award type.",
            },
        },
        "required": ["wants_download"],
    },
}


def _extract_download_intent(question: str, history_block: str = "") -> DownloadIntent | None:
    """Returns None on anything that isn't a clean parse, rather than guessing."""
    system = (
        f"Today's fiscal year is FY{current_fiscal_year()}. Use it for relative phrases "
        "like 'this year' or 'most recent year' if the question doesn't state one explicitly."
    )
    user_content = question
    if history_block:
        system += (
            " The user is continuing a prior download - carry over its agency/award_type from the "
            "history below unless this question names a different one; only the time period usually changes."
        )
        user_content = f"{history_block}\n\nNew question: {question}"
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": _EXTRACT_TOOL_NAME},
            messages=[{"role": "user", "content": user_content}],
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
    if not intent.wants_download:
        return None
    has_year_range = intent.start_year is not None and intent.end_year is not None
    has_date_range = intent.start_date is not None and intent.end_date is not None
    if not has_year_range and not has_date_range:
        return None
    return intent


def _get_status_tolerating_early_404(client: USASpendingClient, file_name: str):
    """A 404 here means the job record isn't indexed yet, not that it doesn't exist - confirmed live."""
    try:
        return client.get_download_status(file_name)
    except USASpendingAPIError as e:
        if str(e).startswith("404"):
            return None
        raise


def _poll_until_finished(client: USASpendingClient, file_name: str):
    """Status moves ready -> running -> finished/failed; both ready and running mean "keep polling"."""
    elapsed = 0.0
    status = _get_status_tolerating_early_404(client, file_name)
    while (status is None or status.status in ("ready", "running")) and elapsed < _POLL_TIMEOUT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS
        status = _get_status_tolerating_early_404(client, file_name)
    if status is None:
        raise USASpendingAPIError(
            f"Download job {file_name} never became queryable within {_POLL_TIMEOUT_SECONDS}s."
        )
    return status


def _build_download_citation(agency_name: str | None, intent: DownloadIntent, filters: AdvancedFilters) -> ToolCitation:
    """Built locally, not via the shared capture-buffer drain - that contextvar set doesn't reliably propagate out of @traceable."""
    params: dict[str, str | int | float] = {"agency_name": agency_name or "all agencies"}
    if intent.start_date and intent.end_date:
        params["start_date"] = intent.start_date
        params["end_date"] = intent.end_date
        period_label = f"{intent.start_date} to {intent.end_date}"
    else:
        params["start_year"] = intent.start_year
        params["end_year"] = intent.end_year
        period_label = f"{year_label(intent.time_period_type, intent.start_year)}-{year_label(intent.time_period_type, intent.end_year)}"
    if intent.award_type:
        params["award_type"] = intent.award_type
    description = f"Award CSV download, {agency_name or 'all agencies'}, {period_label}"
    body = {"filters": filters.model_dump(exclude_none=True), "columns": _DOWNLOAD_COLUMNS, "file_format": "csv"}
    curl = f"curl -X POST '{BASE_URL}/api/v2/download/awards/' -H 'Content-Type: application/json' -d '{json.dumps(body)}'"
    return ToolCitation(tool_name="download_awards", parameters=params, description=description, curl=curl)


def _tag_download_outcome(outcome: str) -> None:
    """Tags whichever run is active in the ask() call stack (there's no
    @traceable on this function itself) with this pilot's own outcome
    vocabulary - a no-op outside a traced call (e.g. streaming.py, which
    has no enclosing @traceable span yet - a separate, pre-existing gap).
    """
    run_tree = get_current_run_tree()
    if run_tree is not None:
        run_tree.add_metadata({"download_outcome": outcome})


def handle_download_request(question: str, conversation_id: str, recent_messages: list | None = None):
    """Returns None to signal "fall through to the normal tool loop unchanged"."""
    from .orchestrator import AgentResult

    unsupported = _unsupported_download_label(question)
    if unsupported is not None:
        _tag_download_outcome("unsupported_endpoint")
        return AgentResult(
            answer_text=(
                f"Downloading {unsupported} isn't supported yet — only award-level CSV "
                "exports are, for now."
            ),
            conversation_id=conversation_id,
        )

    history_block = _render_recent_exchanges(recent_messages) if recent_messages else ""
    intent = _extract_download_intent(question, history_block)
    if intent is None:
        _tag_download_outcome("no_clear_intent")
        return None

    client = _get_usaspending_client()
    agency_name = None
    if intent.agency_raw:
        match = client.find_agency_by_name(intent.agency_raw)
        if match is None:
            _tag_download_outcome("unknown_agency")
            return None
        agency_name = match.agency_name

    try:
        # Placeholder when start_date is set - real scope is applied below by overwriting time_period.
        year_for_filters = int(intent.start_date[:4]) if intent.start_date else intent.start_year
        filters = _build_filters(
            client,
            agency_name,
            intent.time_period_type,
            year_for_filters,
            year_for_filters,
            award_type=intent.award_type,
            scope_required=False,
        )
        if intent.start_date and intent.end_date:
            filters.time_period = [TimePeriod(start_date=intent.start_date, end_date=intent.end_date)]
        job = client.download_awards(filters, _DOWNLOAD_COLUMNS)
        citation = _build_download_citation(agency_name, intent, filters)
        status = _poll_until_finished(client, job.file_name)
    except USASpendingAPIError as e:
        logger.warning("Download pipeline failed for question %r: %s", question, e)
        _tag_download_outcome("api_error")
        return AgentResult(answer_text=f"This download failed: {e}.", conversation_id=conversation_id)

    _tag_download_outcome(status.status)
    download = DownloadSpec(
        file_name=status.file_name, url=status.file_url, status_url=job.status_url,
        status=status.status, total_rows=status.total_rows,
    )
    if status.status == "finished":
        answer_text = (
            f"Your download is ready: {status.total_rows or 0} rows in {status.file_name}.\n{status.file_url}"
        )
    elif status.status == "failed":
        answer_text = f"This download failed to generate: {status.message or 'no further detail from the API.'}"
    else:
        answer_text = (
            f"Your download is still generating ({status.file_name}). It'll appear at the link below "
            "once ready - check back shortly."
        )

    return AgentResult(
        answer_text=answer_text, conversation_id=conversation_id, downloads=[download], tool_citations=[citation]
    )

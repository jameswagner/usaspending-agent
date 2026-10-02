"""Pydantic request/response models for the FastAPI routes in main.py."""
from __future__ import annotations

from pydantic import BaseModel

from backend.app.agent.response_shaping import (
    Citation,
    DownloadSpec,
    FollowUp,
    ToolCitation,
)


class AskRequest(BaseModel):
    question: str
    conversation_id: str | None = None


class AskResponse(BaseModel):
    answer_text: str
    source_type: str
    conversation_id: str
    charts: list[dict] = []
    citations: list[Citation] = []
    tool_citations: list[ToolCitation] = []
    downloads: list[DownloadSpec] = []
    follow_ups: list[FollowUp] = []


class DownloadFollowUpRequest(BaseModel):
    """POST body for /ask/download - the "Download this" button's click path.
    filters is the structured, DownloadIntent-shaped dict a FollowUp (kind="download")
    carried; conversation_id is required (a follow-up only ever appears attached to an
    already-existing turn, unlike AskRequest's conversation_id which starts a thread)."""

    filters: dict[str, str | int | float | list[str]]
    conversation_id: str

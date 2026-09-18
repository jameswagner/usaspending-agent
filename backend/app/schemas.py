"""Pydantic request/response models for the FastAPI routes in main.py."""
from __future__ import annotations

from pydantic import BaseModel

from backend.app.agent.response_shaping import Citation, ToolCitation


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

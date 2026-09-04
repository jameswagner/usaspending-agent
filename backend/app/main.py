"""FastAPI app exposing POST /ask, backed by the tool-calling agent."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from backend.app.agent import NOT_FOUND_MESSAGE, warm_up
from backend.app.agent import ask as agent_ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    yield


app = FastAPI(title="USASpending RAG", lifespan=lifespan)


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer_text: str
    source_type: str
    chart_data: dict | None = None
    citations: list[Citation] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = agent_ask(request.question)
    # The agent's tools return plain text, not structured chunk/agency
    # metadata, so there's no clean citation list to build yet - tracked
    # in BACKLOG.md as a real gap, not silently dropped.
    source_type = "not_found" if result.answer_text == NOT_FOUND_MESSAGE else "agent"
    chart_data = result.chart.model_dump() if result.chart else None
    return AskResponse(
        answer_text=result.answer_text,
        source_type=source_type,
        chart_data=chart_data,
        citations=[],
    )

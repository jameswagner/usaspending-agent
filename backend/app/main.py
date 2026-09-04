"""FastAPI app exposing POST /ask, backed by the tool-calling agent."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from backend.app.agent import NOT_FOUND_MESSAGE, Citation, warm_up
from backend.app.agent import ask as agent_ask


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    yield


app = FastAPI(title="USASpending RAG", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer_text: str
    source_type: str
    charts: list[dict] = []
    citations: list[Citation] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    result = agent_ask(request.question)
    # Citations only cover search_guide (chunk id/source/page) so far - a
    # live API-call citation (query parameters, no "page" to point to) is
    # a separate, deferred piece. See BACKLOG.md.
    source_type = "not_found" if result.answer_text == NOT_FOUND_MESSAGE else "agent"
    return AskResponse(
        answer_text=result.answer_text,
        source_type=source_type,
        charts=[c.model_dump() for c in result.charts],
        citations=result.citations,
    )


# Minimal no-build frontend (plain HTML/JS, no npm) - mounted after the API
# routes above, at /ui rather than "/", so it can't shadow /ask or /health.
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

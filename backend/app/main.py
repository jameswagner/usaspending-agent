"""FastAPI app exposing POST /ask, backed by the tool-calling agent."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

from backend.app.agent import ask as agent_ask
from backend.app.agent.orchestrator import NOT_FOUND_MESSAGE
from backend.app.agent.response_shaping import Citation, ToolCitation
from backend.app.agent.singletons import warm_up
from backend.app.logging_config import configure_logging

logger = logging.getLogger(__name__)

# Requests allowed per client IP per minute, configurable via env var so a
# public deployment can tighten this without a code change. See BACKLOG.md's
# "Rate limiting on /ask" entry: without this, any caller could send
# unlimited requests, each costing a real Claude call plus an uncapped
# number of live USASpending API calls. Keyed on remote address (slowapi's
# get_remote_address) - doesn't look at X-Forwarded-For, so behind a
# reverse proxy every request would look like it comes from the proxy's
# IP; not an issue for direct local/demo use, would need addressing before
# deploying behind one.
ASK_RATE_LIMIT_PER_MINUTE = int(os.environ.get("ASK_RATE_LIMIT_PER_MINUTE", "20"))

limiter = Limiter(key_func=get_remote_address, headers_enabled=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    warm_up()
    yield


app = FastAPI(title="USASpending RAG", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer_text: str
    source_type: str
    charts: list[dict] = []
    citations: list[Citation] = []
    tool_citations: list[ToolCitation] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(f"{ASK_RATE_LIMIT_PER_MINUTE}/minute")
def ask(request: Request, response: Response, payload: AskRequest) -> AskResponse:
    # slowapi's @limiter.limit needs a starlette Request (by convention
    # named "request") to key the limit on the caller's IP, and a
    # starlette Response (by convention named "response") to write
    # X-RateLimit-*/Retry-After headers onto on a *successful* call - since
    # this endpoint returns a plain AskResponse, not a Response object,
    # slowapi has nothing to write headers onto without it. That's why the
    # request body param below is "payload", not "request".
    logger.info("Received question: %r", payload.question)
    try:
        result = agent_ask(payload.question)
    except Exception:
        # Without this, an exception here (a bug in the agent loop, an
        # unhandled API error) would only ever surface as FastAPI's generic
        # 500 response - no application-level record of what actually
        # broke or which question triggered it.
        logger.exception("agent_ask raised for question: %r", payload.question)
        raise
    # citations covers search_guide (chunk id/source/page); tool_citations
    # covers the four live-data tools (tool name + query parameters, since
    # there's no "page" to point to for a live lookup). Kept as two lists
    # rather than one discriminated model since the shapes don't overlap.
    source_type = "not_found" if result.answer_text == NOT_FOUND_MESSAGE else "agent"
    logger.info("Answered question with source_type=%s", source_type)
    return AskResponse(
        answer_text=result.answer_text,
        source_type=source_type,
        charts=[c.model_dump() for c in result.charts],
        citations=result.citations,
        tool_citations=result.tool_citations,
    )


# Minimal no-build frontend (plain HTML/JS, no npm) - mounted after the API
# routes above, at /ui rather than "/", so it can't shadow /ask or /health.
app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")

"""FastAPI app exposing POST /ask, backed by the hybrid dense+BM25 retriever."""
from __future__ import annotations

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from backend.app.generation import synthesize_answer
from backend.app.retrieval.hybrid import (
    HybridRetriever,
)

# Calibrated via dev_tools/calibrate_threshold.py (LLM-generated labeled
# question set, threshold chosen to best separate the two groups): the
# data-driven optimum was -1.89 on that run; using -2.0 for a small safety
# margin rather than the razor-exact value. Previously an eyeballed -5.0,
# which calibration showed was too permissive (let some "sounds relevant
# but needs live data" negatives through).
RERANK_CONFIDENCE_THRESHOLD = -2.0

retriever: HybridRetriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global retriever
    retriever = HybridRetriever()
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
    results = retriever.retrieve(request.question, top_k=3)
    matches = [r for r in results if r["rerank_score"] > RERANK_CONFIDENCE_THRESHOLD]

    if not matches:
        return AskResponse(
            answer_text="I couldn't find anything in the Analyst's Guide that confidently answers this question.",
            source_type="document",
            citations=[],
        )

    answer_text = synthesize_answer(request.question, matches)
    citations = [
        Citation(chunk_id=m["id"], source=m["source"], page=m["page_start"])
        for m in matches
    ]
    return AskResponse(answer_text=answer_text, source_type="document", citations=citations)

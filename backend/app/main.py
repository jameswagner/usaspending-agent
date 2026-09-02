"""FastAPI app exposing POST /ask over the ingested Analyst's Guide chunks.

Retrieval here is a placeholder keyword-overlap search over the chunk
JSONL produced by `backend.app.retrieval.ingest`. It exists so the API
shape (request/response schema, endpoint wiring) is testable end to end
before the real hybrid dense+BM25 retriever and reranker are built.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

CHUNKS_PATH = Path("data/chunks/analysts_guide_chunks.jsonl")

app = FastAPI(title="USASpending RAG")


class Citation(BaseModel):
    chunk_id: str
    source: str
    page: int


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer_text: str
    source_type: str
    chart_data: Optional[dict] = None
    citations: List[Citation] = []


def load_chunks() -> List[dict]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9']+", text.lower()))


def top_matching_chunks(question: str, chunks: List[dict], k: int = 3) -> List[dict]:
    q_tokens = tokenize(question)
    scored = []
    for c in chunks:
        overlap = len(q_tokens & tokenize(c["text"]))
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [c for _, c in scored[:k]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    chunks = load_chunks()
    matches = top_matching_chunks(request.question, chunks)

    if not matches:
        return AskResponse(
            answer_text="No matching content found in the Analyst's Guide.",
            source_type="document",
            citations=[],
        )

    answer_text = "\n\n".join(c["text"] for c in matches)
    citations = [
        Citation(chunk_id=c["id"], source=c["source"], page=c["page_start"])
        for c in matches
    ]
    return AskResponse(answer_text=answer_text, source_type="document", citations=citations)

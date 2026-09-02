"""FastAPI app exposing POST /ask, backed by the hybrid dense+BM25 retriever."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

from backend.app.retrieval.hybrid import HybridRetriever  # noqa: E402 (after load_dotenv)

# Below this, rerank scores tend to mean "nothing relevant" rather than a
# weak match: sanity_check.py showed misspelled/out-of-scope queries
# clustering around -10 to -11, while genuine matches score from positive
# down to roughly -3.
RERANK_CONFIDENCE_THRESHOLD = -5.0

retriever: Optional[HybridRetriever] = None


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
    chart_data: Optional[dict] = None
    citations: List[Citation] = []


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

    answer_text = "\n\n".join(m["text"] for m in matches)
    citations = [
        Citation(chunk_id=m["id"], source=m["source"], page=m["page_start"])
        for m in matches
    ]
    return AskResponse(answer_text=answer_text, source_type="document", citations=citations)

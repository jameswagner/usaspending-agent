# usaspending-rag

A tool-calling assistant for questions about USASpending.gov federal spending data. An LLM agent picks between two kinds of tools per question: retrieval over a static PDF guide for conceptual/definitional questions, and live USASpending API calls for actual numbers.

## Architecture

- **Conceptual questions** ("what is a sub-award?") are answered by hybrid retrieval over the Analyst's Guide to Federal Spending Data: dense embeddings (Chroma) + BM25 keyword search (Whoosh), merged and reranked with a cross-encoder.
- **Live-data questions** ("how much did NSF spend on X?") are answered by calling the real USASpending.gov API.
- A tool-calling agent (Claude, via the Anthropic SDK's tool runner) decides which tool(s) a question needs, including questions that need both.
- A cheap classifier gates obviously out-of-scope questions before the (more expensive) agent loop runs at all.

### Tools available to the agent

| Tool | Answers |
|---|---|
| `search_guide` | Definitions and concepts from the Analyst's Guide, with page citations |
| `lookup_agency` | An agency's basic profile (toptier code, mission, website) |
| `get_spending_by_category` | One agency's spending broken down by NAICS/PSC/sub-agency/etc. for a fiscal year range — charts when 2+ categories come back |
| `get_spending_over_time` | One agency's spending trend across fiscal years/quarters/months — charts when 2+ periods come back |
| `search_awards` | Individual contract/grant/loan records for an agency and fiscal year range |

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID if using an
                        # identity-linked key); LANGSMITH_* is optional, for tracing
```

## Building the retrieval indexes (one-time, or whenever the source PDF changes)

```bash
uv run python -m backend.app.retrieval.pipeline.ingest \
  --pdf data/raw/analyst-guide.pdf --out data/chunks/analysts_guide_chunks.jsonl
uv run python -m backend.app.retrieval.pipeline.vector_index --chunks data/chunks/analysts_guide_chunks.jsonl
uv run python -m backend.app.retrieval.pipeline.bm25_index --chunks data/chunks/analysts_guide_chunks.jsonl
```

## Running

```bash
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

- Browser UI: `http://127.0.0.1:8000/ui/`
- API: `POST /ask` with `{"question": "..."}`, returns `{answer_text, source_type, charts, citations}`
- Health check: `GET /health`

You can also run the agent directly from the CLI, without starting the server:

```bash
uv run python -m backend.app.agent --question "What is a prime award?"
```

## Testing

```bash
uv run pytest -q      # unit tests: chunking, retrieval merge logic, chart/citation
                       # eligibility, API client parsing — no network calls, no API cost
uv run ruff check .
```

## Project layout

```
backend/app/
  main.py                 FastAPI app (POST /ask, GET /health, serves frontend/ at /ui)
  agent/                  Tool-calling agent (package): clients, tool definitions, scope gate,
                            chart/citation logic (response_shaping.py), orchestrator, CLI
  retrieval/
    hybrid.py              Dense+sparse retriever with cross-encoder reranking (used at request time)
    pipeline/               One-off scripts: PDF -> chunks -> indexes
    dev_tools/              Manual scripts: sanity_check.py, calibrate_threshold.py
  tools/usaspending_client.py   Typed client for the live USASpending.gov API
frontend/index.html        Minimal no-build UI (vanilla JS, served by FastAPI, no separate process)
tests/                     Unit tests
BACKLOG.md                 Known gaps and deferred work
```

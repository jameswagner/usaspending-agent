# usaspending-rag

A tool-calling assistant for questions about USASpending.gov federal spending data. An LLM agent picks between two kinds of tools per question: retrieval over two conceptual/definitional sources for term lookups, and live USASpending API calls for actual numbers.

## Architecture

- **Conceptual questions** ("what is a sub-award?", "what is an IDV?") are answered by hybrid retrieval over two sources — the Analyst's Guide to Federal Spending Data (a PDF, Q&A-chunked) and the live USASpending Glossary API (~150 terms, one chunk per term) — combined into one Chroma + Whoosh index: dense embeddings (Chroma) + BM25 keyword search (Whoosh), merged and reranked with a cross-encoder.
- **Live-data questions** ("how much did NSF spend on X?") are answered by calling the real USASpending.gov API.
- A tool-calling agent (Claude, via the Anthropic SDK's tool runner) decides which tool(s) a question needs, including questions that need both.
- A cheap classifier gates obviously out-of-scope questions before the (more expensive) agent loop runs at all.
- The model never does multi-number math (totals, percentages, ratios, before/after change, rankings) in its own prose — it calls one of six typed arithmetic tools, or `code_execution` as a fallback for calculations those six don't cover.

### Tools available to the agent

| Tool | Answers |
|---|---|
| `search_guide` | Definitions and concepts from the Analyst's Guide and the USASpending Glossary, with page or term citations |
| `lookup_agency` | An agency's basic profile (toptier code, mission, website) |
| `get_spending_by_category` | One agency's spending broken down by NAICS/PSC/sub-agency/etc. for a fiscal year range — charts when 2+ categories come back |
| `get_spending_over_time` | One agency's spending trend across fiscal years/quarters/months — charts when 2+ periods come back |
| `search_awards` | Individual contract/grant/loan records for an agency and fiscal year range |
| `sum_values`, `average`, `percentage_of`, `delta`, `ratio`, `rank_values` | Deterministic arithmetic over numbers the tools above already returned — totals, shares, before/after change, cross-entity comparison, ranking |
| `code_execution` | Anthropic's sandboxed Python/Bash fallback for calculations the six typed tools don't cover (e.g. a statistic like standard deviation) |

## Setup

```bash
uv sync
cp .env.example .env   # fill in ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID if using an
                        # identity-linked key); LANGSMITH_* is optional, for tracing
```

## Building the retrieval indexes (one-time, or whenever a source changes)

```bash
uv run python -m backend.app.retrieval.pipeline.ingest \
  --pdf data/raw/analyst-guide.pdf --out data/chunks/analysts_guide_chunks.jsonl
uv run python -m backend.app.retrieval.pipeline.ingest_glossary   # fetches the live Glossary API
uv run python -m backend.app.retrieval.pipeline.vector_index     # both sources into one Chroma collection
uv run python -m backend.app.retrieval.pipeline.bm25_index       # both sources into one Whoosh index
```

## Running

```bash
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

- Browser UI: `http://127.0.0.1:8000/ui/`
- API: `POST /ask` with `{"question": "..."}`, returns `{answer_text, source_type, charts, citations, tool_citations}`
- Health check: `GET /health`

You can also run the agent directly from the CLI, without starting the server:

```bash
uv run python -m backend.app.agent --question "What is a prime award?"
```

## Testing

```bash
uv run pytest -q      # unit tests: chunking, retrieval merge logic, chart/citation
                       # eligibility, arithmetic tools, API client parsing — no network
                       # calls, no API cost
uv run ruff check .
```

## Project layout

```
backend/app/
  main.py                 FastAPI app (POST /ask, GET /health, serves frontend/ at /ui)
  agent/                  Tool-calling agent (package): singletons, tool definitions, arithmetic
                            tools, scope gate, chart/citation logic (response_shaping.py),
                            orchestrator, CLI
    dev_tools/              Manual, opt-in scripts (real billed LLM calls, not in CI):
                              red-team (data-injection, jailbreak, prompt-extraction,
                              resource-abuse) and a code_execution wiring check
  retrieval/
    hybrid.py              Dense+sparse retriever with cross-encoder reranking (used at request time)
    pipeline/               One-off scripts: PDF/Glossary API -> chunks -> indexes
    dev_tools/              Manual scripts: sanity_check.py, calibrate_threshold.py
  usaspending_client.py   Typed client for the live USASpending.gov API
frontend/index.html        Minimal no-build UI (vanilla JS, served by FastAPI, no separate process)
tests/                     Unit tests
BACKLOG.md                 Known gaps and deferred work
```

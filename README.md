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
| `get_agency_budget` | An agency's actual appropriated budgetary resources, obligations, and outlays for a fiscal year range — the real answer to "what is X's budget," as opposed to the three spending tools below, which report award-level spending (a different, non-interchangeable number) |
| `get_spending_by_category` | Spending broken down by NAICS/PSC/sub-agency/etc. for a fiscal year range, scoped by an awarding agency and/or a recipient (at least one required) — optionally filtered further by award type, recipient name/id, amount range, US state (place of performance or recipient location), keywords, `date_type` (action_date/date_signed/last_modified_date/new_awards_only), domestic/foreign scope, or an exact NAICS/PSC/CFDA code — charts when 2+ categories come back |
| `get_spending_over_time` | A spending trend across fiscal years/quarters/months, same agency-and/or-recipient scoping and optional filters as `get_spending_by_category` — charts when 2+ periods come back |
| `search_awards` | Individual contract/grant/loan records for a fiscal year range, scoped by an awarding agency and/or a recipient, ranked largest-amount-first by default, with the same optional filters as `get_spending_by_category` (except `recipient_id`, confirmed silently ignored by the live API on this endpoint). Each result includes an `internal_id` for a follow-up `get_award_details` call. Flags when a result list is truncated (more matches than shown) rather than presenting a partial list as complete |
| `get_award_details` | Full details for one specific award (contract, IDV, grant, loan, or other financial assistance) found via `search_awards` — description, dates, competition data, recipient, funding breakdown, and parent-vehicle linkage. `include_child_orders=True` fetches the real child/grandchild-order rollup for an IDV (contract vehicle), since an IDV's own reported total can show $0 even when it's an active, heavily-used vehicle |
| `search_recipients` | Find a company/organization/individual's exact `recipient_id` by name, UEI, or DUNS — a name alone is often genuinely ambiguous (e.g. "Boeing" resolves to 6+ distinct recipients sharing the same display name), so this shows every real candidate rather than silently picking one, for a precise follow-up via `get_recipient_details` or the `recipient_id` filter above |
| `get_recipient_details` | Full profile for one already-resolved recipient — identity, parent company, address, business types, and total federal transactions for a fiscal year, `"all"` (default), or `"latest"` (trailing 12 months) |
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

Two processes: the FastAPI backend, and the Next.js frontend that talks to it
through a server-side proxy route.

```bash
uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```bash
cd web && npm install   # first time only
npm run dev
```

- Browser UI: `http://localhost:3000`
- API: `POST /ask` with `{"question": "...", "conversation_id": "..."}`, returns
  `{answer_text, source_type, conversation_id, charts, citations, tool_citations}`
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
  main.py                 FastAPI app (POST /ask, GET /health)
  agent/                  Tool-calling agent (package): singletons, tool definitions
                            (tools.py) + their shared filter-building layer
                            (tool_filters.py), arithmetic tools, scope gate,
                            chart/citation logic (response_shaping.py), orchestrator, CLI
    dev_tools/              Manual, opt-in scripts (real billed LLM calls unless noted,
                              not in CI): red-team (data-injection, jailbreak,
                              prompt-extraction, resource-abuse), a code_execution wiring
                              check, live verification of the spending-tool filters
                              (verify_shared_filters.py), and a free/local coverage-diff
                              check against the live API contract (check_filter_coverage.py)
  retrieval/
    hybrid.py              Dense+sparse retriever with cross-encoder reranking (used at request time)
    pipeline/               One-off scripts: PDF/Glossary API -> chunks -> indexes
    dev_tools/              Manual scripts: sanity_check.py, calibrate_threshold.py
  usaspending_client.py   Typed client for the live USASpending.gov API
web/                        Next.js frontend (separate process, proxies to the FastAPI API)
tests/                     Unit tests
BACKLOG.md                 Known gaps and deferred work
private/                   Gitignored: demo script, dev narrative, blog posts - not part of the deliverable
```

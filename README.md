# usaspending-rag

A tool-calling assistant for questions about USASpending.gov federal spending data. An LLM agent picks between retrieval (conceptual/definitional term lookups, and semantic NAICS/PSC/CFDA/county code lookups) and live USASpending API calls (actual numbers), across as many tool calls and turns as a question needs.

## Architecture

- **Conceptual questions** ("what is a sub-award?", "what is an IDV?") are answered by hybrid retrieval over two sources — the Analyst's Guide to Federal Spending Data (a PDF, Q&A-chunked) and the live USASpending Glossary API (~150 terms, one chunk per term) — combined into one Chroma + Whoosh index: dense embeddings (Chroma) + BM25 keyword search (Whoosh), merged and reranked with a cross-encoder.
- **Code-lookup questions** ("what NAICS code is custom software development?") resolve a plain-English description to a NAICS/PSC/CFDA code or county FIPS code, each (except county, which calls the live location endpoint directly) via its own separate Chroma + Whoosh index, same hybrid-retrieval shape as the conceptual index above.
- **Live-data questions** ("how much did NSF spend on X?") are answered by calling the real USASpending.gov API.
- A tool-calling agent decides which tool(s) a question needs, including questions that need more than one. It runs on LangGraph by default (`create_react_agent`, via `langgraph_tools.py`'s LangChain-compatible wrappers of the same tool functions), with a SQLite checkpointer keyed by `conversation_id` giving real multi-turn memory — a follow-up like "what about last year?" resolves against the prior turn's history. `AGENT_ENGINE=legacy` switches to the original, stateless Anthropic SDK tool-runner loop.
- A cheap classifier gates obviously out-of-scope questions before the (more expensive) agent loop runs at all — a first-turn version (bare question + best retrieval passage) and a separate follow-up version that folds in prior conversation turns, so a continuation like "was that a lot?" isn't misjudged as off-topic just because it has no keywords of its own.
- The model never does multi-number math (totals, percentages, ratios, before/after change, rankings) in its own prose — it calls one of six typed arithmetic tools, or `code_execution` as a fallback for calculations those six don't cover.

### Tools available to the agent

| Tool | Answers |
|---|---|
| `search_guide` | Definitions and concepts from the Analyst's Guide and the USASpending Glossary, with page or term citations |
| `lookup_agency` | An agency's basic profile (toptier code, mission, website) |
| `resolve_naics_code` | Plain-English industry/business description → matching NAICS code(s), for the `naics_code` filter below — semantic matches against the official code index, not a confirmed exact lookup |
| `resolve_psc_code` | Same idea for a product/service description → PSC (Product and Service Code) code(s), for the `psc_code` filter |
| `resolve_cfda_program` | Same idea for a federal grant/loan/assistance program description → CFDA/Assistance Listing program number(s), for the `cfda_program` filter |
| `resolve_county_fips` | A county name → its 3-digit FIPS code, for the `performed_in_county`/`recipient_in_county` filters — strips local suffixes (Louisiana parishes, Alaska boroughs/census areas) the live endpoint's own match otherwise fails on |
| `list_top_agencies_by_budget` | Agencies ranked by budget authority, largest first, with each one's share of the total federal budget — always the current fiscal year/quarter, no historical range |
| `get_agency_budget` | An agency's actual appropriated budgetary resources, obligations, and outlays for a fiscal year range — the real answer to "what is X's budget," as opposed to the spending tools below, which report award-level spending (a different, non-interchangeable number) |
| `get_agency_award_breakdown` | One agency's award obligations *and* transaction/new-award counts, broken down by sub-agency, for a single fiscal year — `get_spending_by_category` has no count fields at all |
| `get_spending_by_category` | Spending broken down by NAICS/PSC/sub-agency/etc. for a fiscal year range, scoped by an awarding agency and/or a recipient (at least one required) — optionally filtered further by award type, recipient name/id, amount range, US state (place of performance or recipient location), keywords, `date_type` (action_date/date_signed/last_modified_date/new_awards_only), domestic/foreign scope, or an exact NAICS/PSC/CFDA code — charts when 2+ categories come back |
| `get_spending_over_time` | A spending trend across fiscal years/quarters/months, same agency-and/or-recipient scoping and optional filters as `get_spending_by_category` — charts when 2+ periods come back |
| `get_spending_by_geography` | Spending ranked by state, county, congressional district, or country in one call, instead of checking one place at a time — population and per-capita figures reflect current data, not the queried period |
| `search_awards` | Individual contract/grant/loan records for a fiscal year range, scoped by an awarding agency and/or a recipient, ranked largest-amount-first by default, with the same optional filters as `get_spending_by_category` (except `recipient_id`, confirmed silently ignored by the live API on this endpoint). Each result includes an `internal_id` for a follow-up `get_award_details` call. Flags when a result list is truncated (more matches than shown) rather than presenting a partial list as complete |
| `get_award_details` | Full details for one specific award (contract, IDV, grant, loan, or other financial assistance) found via `search_awards` — description, dates, competition data, recipient, funding breakdown, and parent-vehicle linkage. `include_child_orders=True` fetches the real child/grandchild-order rollup for an IDV (contract vehicle), since an IDV's own reported total can show $0 even when it's an active, heavily-used vehicle |
| `search_subawards` | Individual subaward records — money a prime awardee passed on to a sub-recipient — scoped by an awarding agency and/or a sub-recipient. `recipient_name` and every `recipient_in_*` location parameter filter the *sub*-recipient here, the opposite of what those same names mean on every other spending tool above, which filter the prime |
| `get_award_subawards` | The complete subaward list for one specific prime award already found via `search_awards`, given its `internal_id` — the award-profile page's own Sub-Awards tab, as opposed to `search_subawards`' cross-award search |
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

The conceptual/definitional index (Guide + Glossary):

```bash
uv run python -m backend.app.retrieval.pipeline.ingest \
  --pdf data/raw/analyst-guide.pdf --out data/chunks/analysts_guide_chunks.jsonl
uv run python -m backend.app.retrieval.pipeline.ingest_glossary   # fetches the live Glossary API
uv run python -m backend.app.retrieval.pipeline.vector_index \
  --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl
uv run python -m backend.app.retrieval.pipeline.bm25_index \
  --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl
```

The three code-lookup indexes (`resolve_naics_code`/`resolve_psc_code`/`resolve_cfda_program`), each
its own Chroma collection/Whoosh index via env var overrides so they don't collide with the one above
or each other — source files already in `data/raw/` (Census NAICS files, the PSC manual, the SAM.gov
CFDA bulk CSV):

```bash
uv run python -m backend.app.retrieval.pipeline.ingest_naics
uv run python -m backend.app.retrieval.pipeline.ingest_psc
uv run python -m backend.app.retrieval.pipeline.ingest_cfda

CHROMA_DB_DIR=./data/chroma_naics WHOOSH_INDEX_DIR=./data/whoosh_naics \
  uv run python -m backend.app.retrieval.pipeline.vector_index --chunks data/chunks/naics_chunks.jsonl
CHROMA_DB_DIR=./data/chroma_naics WHOOSH_INDEX_DIR=./data/whoosh_naics \
  uv run python -m backend.app.retrieval.pipeline.bm25_index --chunks data/chunks/naics_chunks.jsonl
# same pattern for psc (chroma_psc/whoosh_psc, psc_chunks.jsonl) and cfda (chroma_cfda/whoosh_cfda, cfda_chunks.jsonl)
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
  `{answer_text, source_type, conversation_id, charts, citations, tool_citations}` —
  omit `conversation_id` on the first call, then pass back the one returned to continue
  the same multi-turn thread
- Rate limited to `ASK_RATE_LIMIT_PER_MINUTE` (default 20) requests/minute per client IP
- Health check: `GET /health`

You can also run the agent directly from the CLI, without starting the server:

```bash
uv run python -m backend.app.agent --question "What is a prime award?"
```

## Testing

```bash
uv run pytest -q      # unit tests: chunking/ingestion (Guide, Glossary, NAICS, PSC,
                       # CFDA), retrieval merge logic, chart/citation eligibility,
                       # arithmetic tools, API client parsing, conversation persistence,
                       # the LangGraph tool wrappers — no network calls, no API cost
uv run ruff check .
```

`agent/dev_tools/` and `retrieval/dev_tools/` hold separate, opt-in scripts (real billed
LLM calls for most of them) not run by the above or by CI — see Project layout below.

## Project layout

```
backend/app/
  main.py                  FastAPI app (POST /ask, GET /health)
  logging_config.py        Shared logging setup (server + CLI)
  agent/                   Tool-calling agent (package)
    singletons.py            Lazily-constructed clients: Anthropic/LangChain chat
                                models, the Guide+Glossary and NAICS/PSC/CFDA
                                retrievers, the USASpending API client, and the
                                LangGraph checkpointer + conversation graph
    tools/                   The 18 @beta_tool data tools, split by concern:
                                _shared.py (call recording, per-turn budget,
                                untrusted-data wrapping, + search_guide/
                                lookup_agency/get_agency_budget/
                                list_top_agencies_by_budget/
                                get_agency_award_breakdown), spending.py
                                (get_spending_by_category/get_spending_over_time/
                                search_awards/search_subawards/
                                get_spending_by_geography, all funneled through
                                tool_filters._build_filters), awards.py
                                (get_award_details/get_award_subawards),
                                recipients.py (search_recipients/
                                get_recipient_details) + business_type_labels.py
                                (its code->label table), naics.py/psc.py/cfda.py/
                                location.py (the resolve_*_code / resolve_county_fips
                                semantic code-lookup tools)
    tool_filters.py           Shared filter-building layer for the spending tools
    arithmetic_tools.py       The six deterministic arithmetic tools
    langgraph_tools.py        LangChain-compatible wrappers of the same tool
                                functions, for the LangGraph agent path
    scope.py                  In-scope gate: a first-turn classifier plus a
                                separate follow-up classifier that folds in prior
                                conversation turns
    response_shaping.py       Chart/citation logic, fiscal-year math
    orchestrator.py           System prompt, AgentResult, ask() - dispatches to
                                the LangGraph path (default) or the legacy
                                stateless tool-runner path (AGENT_ENGINE=legacy)
    cli.py                    The --question CLI entry point
    dev_tools/                Manual, opt-in scripts (real billed LLM calls unless
                                noted, not in CI): red-team checks (data-injection,
                                jailbreak, prompt-extraction, resource-abuse),
                                tool-selection and code-lookup accuracy evals
                                (LangSmith Dataset + evaluate()), scope-classifier
                                calibration, live 3-turn conversation-path
                                verification, a code_execution wiring check, live
                                verification of the spending-tool filters
                                (verify_shared_filters.py), a free/local
                                coverage-diff check against the live API contract
                                (check_filter_coverage.py), and a LangSmith
                                trace-dump utility (fetch_trace.py)
  retrieval/
    hybrid.py                 Dense+sparse retriever with cross-encoder reranking (used at request time)
    pipeline/                  One-off scripts: Guide PDF/Glossary API/NAICS/PSC/CFDA source data -> chunks -> indexes
    dev_tools/                 Manual scripts: sanity_check.py, calibrate_threshold.py
  usaspending_client.py    Typed client for the live USASpending.gov API
web/                         Next.js frontend (separate process; proxies to the
                                FastAPI API through web/src/app/api/ask/route.ts)
tests/                       Unit tests
BACKLOG.md                   Known gaps and deferred work
private/                     Gitignored: demo script, dev narrative, blog posts - not part of the deliverable
```

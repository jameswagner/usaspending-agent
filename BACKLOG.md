# Backlog

Deferred ideas and known minor issues — not urgent, not forgotten.

## Daily health check for USASpending API category support

`get_spending_by_category` in `backend/app/agent.py` hardcodes a list of 14
verified-working categories out of the 18 the API contract documents (4 —
`object_class`, `program_activity`, `recipient_parent_duns`, `tas` — 404 live
despite being in the docs, checked 2026-09-03). The tool's error handling
adapts to the live response regardless, so a stale list only costs an
occasional wasted round-trip, never a wrong answer — this isn't correctness-
critical.

Idea: a scheduled job that re-verifies all 18 categories against the live API
daily, persists the result (a JSON file is probably enough), and the tool
reads that instead of the hardcoded list. Needs a scheduler + a persistence
layer we don't have yet — worth doing if this becomes more than a side
project, not before.

## POST /ask returns empty citations

`chart_data` is now wired up (2026-09-04): `get_spending_by_category` and
`get_spending_over_time` each split into a `..._raw()` function (structured
Pydantic response) and the `@beta_tool` wrapper (formats it for the LLM,
unchanged behavior). `should_chart(tool_name, structured_result) -> ChartSpec | None`
picks bar/line charts by result cardinality. A `contextvars.ContextVar`-based
capture buffer records each spending tool's structured result during
`ask()`'s tool-calling loop, isolated per request (verified live: two
concurrent /ask calls with different chart-worthy questions each got back
their own chart, no cross-contamination) — `ask()` now returns an
`AgentResult` (`answer_text` + optional `chart`), and `main.py` reads
`.chart` into the response's `chart_data`. If a turn calls more than one
chart-worthy tool, the first one found is used — a deliberate, named
simplification, not a silent default.

`citations` is still always empty, and remains a separate, deferred gap:
the agent's tools return plain text to the LLM, not structured source
metadata, so there's no clean list to build `AskResponse.citations` from.
Idea: have each tool return structured data including source metadata, and
have `agent.ask()` walk the tool_runner's message history (or the same
capture-buffer mechanism used for charts) to collect which sources were
actually used across the turn. Real work, not a quick fix.

## Tied rerank scores in sanity_check.py

The `NAICS` query in `backend/app/retrieval/dev_tools/sanity_check.py` has two results
tied at the exact same rerank score (4.69). Noticed during hybrid retriever
review, never investigated. Probably harmless, worth a second look sometime.

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

## POST /ask: chart_data done, guide citations done, live-data citations still open

`chart_data` (2026-09-04): `get_spending_by_category` and `get_spending_over_time`
each split into a `..._raw()` function (structured Pydantic response) and the
`@beta_tool` wrapper (formats it for the LLM, unchanged behavior).
`should_chart(tool_name, structured_result) -> ChartSpec | None` picks bar/line
charts by result cardinality. A `contextvars.ContextVar`-based capture buffer
records each spending tool's structured result during `ask()`'s tool-calling
loop, isolated per request (verified live: two concurrent /ask calls with
different chart-worthy questions each got back their own chart, no cross-
contamination). If a turn calls more than one chart-worthy tool, the first
one found is used — a deliberate, named simplification, not a silent default.

Guide citations (2026-09-04): `search_guide` now records its matched chunks
into the same capture buffer; `ask()` builds `Citation(chunk_id, source, page)`
from any `search_guide` calls in the turn, deduped by chunk id. Verified live
via CLI and the running server.

**Still open — live-data citations (lookup_agency, the three spending tools):**
there's no "page" for a live API call, but the analog is straightforward:
cite the tool name + exact query parameters used (agency, category, date
range) — fully reproducible, and the data's already sitting in the same
capture buffer used for charts/guide citations, so this is mechanical, not
a design problem.

**Genuinely unsolved, and a different kind of problem:** citing the raw data
behind a spending answer doesn't verify any arithmetic the model does *on
top of* that data in prose (e.g. "these top two categories account for over
$458 million" — a sum the model computed, not a number the API returned).
A citation to the underlying query would vouch for the raw numbers being
real, not for whether the model added them correctly. The fix that fits how
this project has generally handled model-trust problems (code-enforced
correctness over prompt-trust, e.g. the agency-name-resolution fix) would be
to stop letting the model compute aggregates freely - have the tool's own
formatting code pre-compute verified totals for the model to reference
instead - but that's a bigger behavior change, not attempted yet.

## Tied rerank scores in sanity_check.py

The `NAICS` query in `backend/app/retrieval/dev_tools/sanity_check.py` has two results
tied at the exact same rerank score (4.69). Noticed during hybrid retriever
review, never investigated. Probably harmless, worth a second look sometime.

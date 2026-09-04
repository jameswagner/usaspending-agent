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

`main.py` now calls the tool-calling agent (`backend/app/agent.py`) instead of
the old retrieve-then-synthesize flow, but the agent's tools (`search_guide`,
`lookup_agency`, `get_spending_by_category`) return plain text, not structured
chunk/agency metadata — so there's no clean list to build `AskResponse.citations`
from anymore. It's returned empty rather than faked.

Idea: have each tool return structured data (text + source metadata) instead
of a plain string, and have `agent.ask()` walk the tool_runner's message
history to collect which sources were actually used across the turn. Real
work, not a quick fix — deferred rather than done half-right.

## Tied rerank scores in sanity_check.py

The `NAICS` query in `backend/app/retrieval/dev_tools/sanity_check.py` has two results
tied at the exact same rerank score (4.69). Noticed during hybrid retriever
review, never investigated. Probably harmless, worth a second look sometime.

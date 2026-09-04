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

## POST /ask returns empty citations AND null chart_data — in-progress state

`main.py` calls the tool-calling agent (`backend/app/agent.py`) instead of
the old retrieve-then-synthesize flow, but the agent's tools return plain
text to the LLM, not structured metadata — so there's no clean list to
build `AskResponse.citations` from, and `chart_data` is always `None`.

**Current state of the chart_data half (as of 2026-09-04):** `get_spending_by_category`
and `get_spending_over_time` are each split into a `..._raw()` function
(calls the API once, returns the structured Pydantic response) and the
`@beta_tool`-decorated wrapper (formats that into the string the LLM sees,
behavior unchanged). `should_chart(tool_name, structured_result) -> ChartSpec | None`
exists in `agent.py`, is unit tested (`tests/test_agent.py`), and is
verified live to produce correct bar/line specs from real API data.

**Not yet done, deliberately staged as a separate decision:** nothing
in `agent.ask()`'s tool-calling loop captures which `_raw()` result
corresponds to which tool call as the loop executes, and nothing in
`main.py` reads a captured result to populate `chart_data`. The
mechanism for "capture the structured result alongside the string result
during the loop, make it available after the loop ends" hasn't been
designed yet — needs its own discussion before implementing, since it
touches the tool-calling loop shared by every tool, not just the two
chart-relevant ones.

The citations half (structured source metadata, not just chart data) is
a related but separate deferred idea: have each tool return structured
data including source metadata, and have `agent.ask()` walk the
tool_runner's message history to collect which sources were actually
used across the turn. Real work, not a quick fix — deferred rather than
done half-right.

## Tied rerank scores in sanity_check.py

The `NAICS` query in `backend/app/retrieval/dev_tools/sanity_check.py` has two results
tied at the exact same rerank score (4.69). Noticed during hybrid retriever
review, never investigated. Probably harmless, worth a second look sometime.

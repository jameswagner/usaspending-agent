# Backlog

Deferred ideas and known minor issues — not urgent, not forgotten.

## Daily health check for USASpending API category support

`get_spending_by_category` in `backend/app/agent/tools.py` hardcodes a list of 14
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

## POST /ask: chart_data, guide citations, and live-data citations all done

`chart_data` (2026-09-04): `get_spending_by_category` and `get_spending_over_time`
each split into a `..._raw()` function (structured Pydantic response) and the
`@beta_tool` wrapper (formats it for the LLM, unchanged behavior).
`should_chart(tool_name, structured_result) -> ChartSpec | None` picks bar/line
charts by result cardinality. A `contextvars.ContextVar`-based capture buffer
records each spending tool's structured result during `ask()`'s tool-calling
loop, isolated per request (verified live: two concurrent /ask calls with
different chart-worthy questions each got back their own chart, no cross-
contamination). Originally took only the first chart-worthy result per turn;
real UI usage (comparing two agencies' trends) showed that silently dropping
the second one, so `AgentResult.charts` is now a list — one chart per
chart-worthy tool call, each titled with its agency name so multiple charts
in one answer are distinguishable.

Guide citations (2026-09-04): `search_guide` now records its matched chunks
into the same capture buffer; `ask()` builds `Citation(chunk_id, source, page)`
from any `search_guide` calls in the turn, deduped by chunk id. Verified live
via CLI and the running server.

Live-data citations (2026-09-04): `lookup_agency`, `get_spending_by_category`,
`get_spending_over_time`, and `search_awards` now all record a context dict
(tool params — agency name, category, fiscal year range, group, award type)
into the same capture buffer, and `ask()` builds one `ToolCitation` per call
via a new pure function, `build_tool_citation(tool_name, context)` (same role
`should_chart` plays for charts — unit-tested the same way, see
`TestBuildToolCitation`). Design choice made deliberately: `ToolCitation` is
a *separate* Pydantic model from `Citation` (chunk_id/source/page), not a
shared/discriminated shape — there's no chunk id or page number for a live
API call, and forcing one in would mean fake/null values. `AgentResult` and
`AskResponse` carry two separate lists, `citations` and `tool_citations`;
this serializes cleanly through FastAPI's `response_model` (no union/
discriminator needed) and the frontend renders both under one "Sources"
list, guide entries as "source, page N" and tool entries as their
`description` (e.g. "naics breakdown, National Science Foundation,
FY2023-FY2023"). Tool citations dedup by (tool_name, sorted params) — exact
duplicate calls collapse, but two calls to the same tool with different
params (e.g. comparing two agencies' trends) each keep their own citation;
verified live that this doesn't over-merge. `lookup_agency` previously
didn't call `_record_tool_call` at all, and `search_awards` called it with
no context — both fixed as part of this.

Found while wiring the frontend and fixed inline (not a separate bug report,
since it's in code being added by this same change, not pre-existing code):
`ToolCitation.description` is built from tool-call parameters, which trace
back to arguments the model chose — themselves downstream of user-supplied
question text, i.e. a prompt-injection path. The frontend was inserting
citation text into the DOM via `innerHTML`; without escaping, a question
engineered to make the model pass an agency_name like `<img
src=x onerror=...>` as a tool argument would have been a stored-XSS vector.
Added an `escapeHtml()` helper in `frontend/index.html` and applied it to
all interpolated citation fields (including the pre-existing guide
`source`/`page` fields, for consistency).

Verified live via CLI (`lookup_agency`, each spending tool individually, and
a two-agency comparison question that produces two distinct
`get_spending_over_time` citations) and via the running server + `/ask`
(response validates through `AskResponse`, `/ui` serves the updated JS).

## Fixed: off-by-one-fiscal-year bug in date-range tools

Found via real UI usage (2026-09-05): asked to compare two agencies'
spending trends "from 2021 to 2024," the model computed
`start_date="2021-10-01"` for `get_spending_over_time` — the start of
FY2022, not FY2021 (federal fiscal years are named by the year they END
in). Confirmed by inspecting the actual `tool_use` block the model
produced, not guessed.

Fixed by removing the model's ability to get this wrong at all:
`get_spending_by_category`, `get_spending_over_time`, and `search_awards`
now take `start_fiscal_year`/`end_fiscal_year` as integers, and
`fiscal_year_to_date_range()` computes the exact date bounds in code. The
model only identifies which fiscal years are being asked about, not
computes a date boundary. `_format_time_period` also labels periods
explicitly ("FY2021"/"CY2021") rather than a bare year number, so the
distinction survives into chart labels and tool output regardless of the
model's prose. Regression-tested (`tests/test_agent.py`) and verified live
against the exact reproduction question. Left here as a resolved entry
rather than deleted, since the pattern (code-enforced correctness over
trusting model arithmetic) is the same one still open for the "verifying
retrieved-number arithmetic" problem above, and it's a useful example
precedent for whoever tackles that one.

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

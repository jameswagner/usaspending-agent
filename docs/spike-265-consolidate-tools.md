# Spike #265: consolidating the six spending tools

Spike, not an implementation PR — nothing here is wired into `ask()`.
`backend/app/agent/tools/spending/consolidated.py` prototypes both
candidate shapes but is not imported by `langgraph_tools.py`.

## Token cost (real `bind_tools()` payload, real stub functions)

```
current six, real bind_tools() payload:
   4,716  search_awards
   3,654  get_spending_by_category
   3,156  get_spending_over_time
   2,819  get_spending_by_geography
   2,810  search_transactions
   2,656  search_subawards
  19,811  TOTAL

Shape A (one tool, query_spending):            3,221 tokens  -84%
Shape B (two tools, records + aggregate):       4,758 tokens  -76%
```

Both beat the issue's original estimate (5,091/74% and 7,756/61%,
themselves already spliced from real per-parameter schema fragments) —
real stub functions with tighter docstrings than the union-of-fragments
estimate assumed. Six-tool block is 64% of the current 31,053-token
tool-schema array; Shape A cuts the whole array by 53%, Shape B by 48%.

Measured via `backend/app/agent/langgraph_tools.py`'s real
`LANGGRAPH_TOOLS` + `convert_to_anthropic_tool()` path, and the same path
applied to `query_spending`/`search_spending_records`/`aggregate_spending`
from `consolidated.py` — not JSON fragments spliced together.

## Design: dispatch layer

Both shapes route `level`/`group_by` to the six *existing* `@beta_tool`
functions (`search_awards`, `search_subawards`, `search_transactions`,
`get_spending_by_category`, `get_spending_over_time`,
`get_spending_by_geography`) rather than re-deriving their
`_build_filters`/`_record_optional_filter_context`/`_scope_label`/
formatting/error-handling logic. This keeps the spike's own bug surface
confined to the routing decision being measured, at the cost of one extra
Python call per tool invocation versus calling the `_raw` functions
directly — a real implementation PR should decide whether that's worth
collapsing further.

`level: "award" | "subaward" | "transaction"` maps to `search_awards` /
`search_subawards` / `search_transactions`. `group_by` is `None` for row
results, or a category name (matching `get_spending_by_category`'s
`VALID_CATEGORIES`), `"time"` (→ `get_spending_over_time`, with a separate
`time_grouping` param for period size), or `"geography"` (→
`get_spending_by_geography`, requiring `scope`/`geo_layer`).

## Accuracy: curated 22-case comparison

`eval_tool_selection.py`'s dataset sync/`evaluate()` depend on LangSmith,
whose monthly trace quota (6,000) was already exhausted mid-spike by a
concurrent session's baseline run. Scoring itself runs client-side
(`predict()` + the evaluator functions), so this didn't block grading —
just LangSmith's own UI trace view for that run. Built a standalone
harness instead (`spike_harness.py`, not committed — scratch-only) that:

- Bypasses `ask()`/`create_react_agent`/the pre-filter scope gate/chart
  building — a minimal tool-calling loop, so it's **not** identical to the
  production path. It *is* identical across all three conditions below
  (current six / Shape A / Shape B), so the relative comparison is valid
  even though the absolute baseline number differs slightly from the
  production `ask()`-based 89.8%/98-case run.
- Relabeled a curated 22-case subset of `tool_selection_labeled_set.json`
  (not the full ~46 cases that reference the six tools primarily, nor the
  issue's own ~15-20 estimate — a deliberate sample spanning 11 of 12
  affected categories, including all 3 known pre-existing failures from
  the real baseline run) to each shape's tool name(s) and
  `level`/`group_by` arguments.
- No LangSmith calls at all — `Example`/`Run` constructed locally,
  evaluators called directly, real billed Anthropic calls only.

Results (real billed calls, single run, n=22):

| | tool_selection_correct | tool_order (5 multi-step) | tool_args_correct | answer_correctness_judged (15 flagged) |
|---|---|---|---|---|
| Current six (mini-harness) | 86.4% (19/22) | 60% (3/5) | 100% (8/8) | 73.3% |
| Shape A (`query_spending`) | 100% (22/22) | 100% (5/5) | 100% (19/19) | 73.3% |
| Shape B (two tools) | 95.5% (21/22) | 80% (4/5) | 89.5% (17/19) | 86.7% |

The mini-harness baseline reproduced the same 3 failures the real
98-case/89.8%-overall production run found (NAICS/PSC follow-up chains
routing to `get_spending_over_time`; NSF-cybersecurity routing the same
way) — evidence the harness is a faithful enough proxy for this
comparison, not just a different, incomparable measurement.

### Caveats that qualify the headline numbers

**Shape A's 100% partly reflects lost evaluator resolution, not only a
real fix.** Once all six tools collapse to one name, `tool_selection_correct`/
`confusable_alternative_called` can no longer distinguish "called the
right tool" from "called the only tool" for cases whose original
`expected_args`/`acceptable_tools` didn't pin `level`/`group_by`. Some of
the gain is genuine (the NAICS/PSC cases now correctly resolve the code
*and* land in a valid route), but the ambiguous-intent cases
(`acceptable_tools`-only) can't regress by construction post-collapse —
that's a real structural property of Shape A, not an eval artifact to
dismiss, but it does mean Shape A's number overstates how much *better*
the model got at choosing.

**Shape B surfaced one genuine new confusability**: a
congressional-district geography question likely got `group_by="district"`
(a category dimension, mirroring `get_spending_by_category`'s own
`district` category) instead of `group_by="geography"` +
`geo_layer="district"` — two different routes to a similar-sounding
result, an ambiguity this spike's own `GroupBy` enum design introduced by
overloading "district" as both a category value and (via `geography`) a
`geo_layer` value. Doesn't exist in the current six-tool split, where
`get_spending_by_category` and `get_spending_by_geography` are separate
tools with separate parameter names. A real implementation should
disambiguate this (e.g. drop the bare `district` category value in favor
of always requiring `group_by="geography"` + `geo_layer="district"` for
that specific breakdown, or rename one of them).

**Shape B still has the PSC-resolve-skip failure** (called
`aggregate_spending` twice without ever calling `resolve_psc_code`) — a
different failure mode than baseline's (which picked the wrong tool
outright), but still a miss on the same underlying case.

**`answer_matches_tool_output`** (deterministic figure-hallucination
check) was low across all three conditions (20-33%) — this tracks a
pre-existing figure-formatting mismatch (e.g. "$1.2 billion" in the answer
vs. a raw dollar figure in tool output) unrelated to tool consolidation,
not a regression introduced here.

**Sample size**: 22 cases, one run each, no repeat-trial variance
measurement. Real production go/no-go should use the full labeled-set
subset (~46 cases actually reference the six tools primarily, once
`confusable_with`-only references are excluded) through the real `ask()`
path, not this spike's minimal harness.

## Go/no-go

**Go, with conditions** — proceed toward an implementation PR, but not
with this spike's code verbatim:

1. **Prefer Shape B over Shape A.** Shape A's token ceiling is higher
   (84% vs 76%), but Shape B keeps the row-vs-aggregate distinction at the
   tool-name level, which is exactly the piece of judgment the issue
   itself identified as worth preserving — and this spike's own results
   back that reasoning: Shape A's evaluator resolution loss makes its
   accuracy number less trustworthy than Shape B's, and Shape B's
   `tool_order`/`answer_correctness` numbers were as good or better than
   baseline despite that resolution being retained.
2. **Fix the `district` ambiguity** in `GroupBy` before shipping — this
   spike's own schema design, not a pre-existing issue, but a real one a
   production version needs to close.
3. **Re-run against the full affected-case set** (not just this spike's
   curated 22) **through the real `ask()` path** before cutover — this
   spike deliberately used a narrower, faster harness to keep the
   go/no-go decision unblocked by the LangSmith quota outage; a shipping
   decision shouldn't rest on the narrower harness alone.
4. Decide whether the dispatch layer should call the six `_raw` functions
   directly (saves one Python call layer, but re-introduces the
   formatting/error-handling duplication this spike deliberately avoided)
   or keep delegating to the existing `@beta_tool` wrappers as prototyped
   here.

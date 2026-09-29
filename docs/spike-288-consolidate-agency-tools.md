# Spike #288: consolidating the agency-profile tools

Spike, not an implementation PR — nothing here is wired into `ask()`.
`backend/app/agent/tools/agency_consolidated.py` prototypes both candidate
`group_by`-style tools but is not imported by `langgraph_tools.py` or
`tools/__init__.py`. The three new client methods/models it depends on
(`get_agency_obligations_by_award_category`, `get_agency_sub_component_
federal_accounts`, `get_agency_awards`) are real, live-verified additions
to `client.py`/`models.py` — #286 and #287's gaps, plus the
previously-unused `/agency/{code}/awards/` endpoint — not stubs, per the
issue's own instruction to implement them as real `group_by` branches.

## Two consolidated tools prototyped

**`query_agency_budget(agency_name, start_fiscal_year, end_fiscal_year, group_by=None|"sub_component", bureau=None, include_period_breakdown=False)`**
- `group_by=None` → `get_agency_budget_raw` (range of FYs)
- `group_by="sub_component"`, no `bureau` → `get_agency_budget_by_subcomponent_raw` (single FY)
- `group_by="sub_component"` + `bureau` → new `get_agency_sub_component_federal_accounts` (#287)

**`query_agency_award_activity(agency_name, fiscal_year, group_by=None|"sub_agency"|"award_category", award_type=None, include_offices=False)`**
- `group_by=None` → new `get_agency_awards` (whole-agency totals, previously unused by any tool)
- `group_by="sub_agency"` → `get_agency_award_breakdown_raw`
- `group_by="award_category"` → new `get_agency_obligations_by_award_category` (#286)

`lookup_agency` and `list_top_agencies_by_budget` stay standalone, per the issue.

## Design: dispatch layer

Both tools route to the *existing* `_raw` functions (`get_agency_budget_raw`,
`get_agency_budget_by_subcomponent_raw`, `get_agency_award_breakdown_raw`)
plus the three new client methods directly, reusing
`client.find_agency_by_name` rather than re-deriving agency resolution —
same reasoning #265's spike used for its own dispatch layer (confine the
spike's bug surface to the routing decision being measured).

**Range-vs-single-FY tension, resolved explicitly**: `group_by="sub_component"`
only accepts `start_fiscal_year == end_fiscal_year` — a mismatched range
returns an error string telling the model to call again per year, rather
than silently truncating to one year or ignoring the range. Verified this
actually fires (see the range-rejection case below).

**Bureau resolution** (`group_by="sub_component"` + `bureau=`): the model
passes a bureau name (e.g. "Food and Nutrition Service"), which the
dispatch layer resolves to its `bureau_slug` via a `get_agency_sub_components`
call first (case-insensitive substring match against the parent list),
then calls the new endpoint with the resolved slug — the model is never
expected to know `bureau_slug` values itself.

## Token cost (real `bind_tools()` payload, real functions — not JSON fragments)

Measured via `LANGGRAPH_TOOLS`'s actual `convert_to_anthropic_tool()` path
(same method `d864033` used for #265) plus Anthropic's `count_tokens`
endpoint, for the whole 5-tool (or 2+2) block measured together (not summed
per-tool counts, which double-count fixed per-request overhead):

```
Current 5 agency tools, together:                              2,475 tokens
Consolidated: query_agency_budget + query_agency_award_activity
              + lookup_agency + list_top_agencies_by_budget:    2,034 tokens  (-18%)
No-consolidation: current 5 + 2 new standalone tools for
              #286/#287 (no group_by, separate tools):          2,978 tokens  (+20%)
```

This is a much smaller win than #265's spending-tool consolidation
(76-84% reduction) — expected, and consistent with the issue's own
docstring word-count estimate: the agency tools don't share one large
duplicated filter object the way the six spending tools did. The real
saving here is avoiding the incremental cost of #286/#287 as two more
standalone tools (+20%) rather than shrinking today's baseline
dramatically. Consolidating still beats doing nothing and beats bolting on
two more flat tools, but by a modest margin, not a decisive one.

## Accuracy: standalone harness, real billed calls (Haiku, this app's `AGENT_MODEL` default)

Same reasoning as #265's spike: the prototype isn't wired into `ask()`, so
`eval_tool_selection.py`'s `evaluate()` (which drives the real production
`ask()` path) can't exercise it. Built a minimal tool-calling loop instead
(not committed, scratch-only) that:

- Reuses the *actual* evaluator functions from `eval_tool_selection.py`
  (`tool_selection_correct`, `tool_order_correct`, `tool_args_correct`,
  `confusable_alternative_called`) against locally-constructed `Run`/
  `Example` stand-ins, rather than re-deriving scoring logic.
- Relabeled all ~20 cases in `tool_selection_labeled_set.json` that
  primarily reference the five agency tools (categories: `time_breakdown`,
  `budget_vs_spending`, `agency_award_breakdown`,
  `agency_budget_by_subcomponent`, `agency_budget_ranking`,
  `guide_and_lookup_tools`, `error_paths`) to the consolidated tool names
  + expected `group_by` args, run through both the current 5-tool set and
  the 2-consolidated-tool set for a direct comparison.
- Added 3 new cases with no baseline equivalent (baseline structurally
  cannot answer them: a bureau drill-down, an award-category breakdown,
  and a group_by="sub_component" range-rejection check), run only against
  the consolidated set.
- `query_spending` (the #265 tool) was excluded from both tool sets — it's
  a real acceptable alternative for one case (`time_breakdown`) in
  production, but out of scope for a harness testing only the
  agency-tool surface; dropping it from `acceptable_tools` there doesn't
  change the pass/fail outcome (the agency tool still gets called
  correctly in both arms) but means the harness doesn't fully replicate
  every acceptable path.

Results (real billed calls, single run):

| | tool_selection_correct | tool_args_correct | confusable_alternative_called |
|---|---|---|---|
| Current 5 tools (n=20) | 100% (20/20) | 100% (3/3 graded) | 0% (0/20) |
| Consolidated 2 tools (n=20) | 100% (20/20) | 100% (8/8 graded) | 0% (0/20) |
| Consolidated-only new-gap cases (n=3) | 100% (3/3) | 100% (3/3) | 0% (0/3) |

No regression, and the two new endpoints route correctly on the first try,
including the range-rejection case (the model split a multi-year
sub-component request into 3 separate single-FY calls, exactly the
fallback the error message asks for) and the bureau drill-down (correctly
resolved "Food and Nutrition Service" to its slug and called the new
endpoint, not the flat sub-components list).

One genuine improvement surfaced by the new `group_by=None` branch on
`query_agency_award_activity`: several baseline cases ("How many new
grant awards did NIH issue in FY2024?", "How many transactions did DOE
have in FY2023?") ask a whole-agency question, but `get_agency_award_breakdown`
has no whole-agency mode today — it always returns a by-sub-agency
breakdown, the only option that existed. The consolidated tool now has a
real whole-agency answer (the previously-unused `/agency/{code}/awards/`
endpoint) and the model picked it correctly for these cases rather than
being forced into a sub-agency breakdown it didn't ask for.

### Caveats

- **n=20/n=3, single run, one model (Haiku).** No repeat-trial variance
  measurement, and Haiku is a comparatively easy model for tool selection
  on a small, low-overlap tool surface — a harder tool surface or a
  smaller/weaker model might show separation this run didn't. A real
  go/no-go should re-run on the full labeled set through the real `ask()`
  path (Sonnet, the model `ask()` actually uses by default in production
  configs) before cutover.
- **`answer_correctness_judged` was not run this round** (scope/cost
  reduction vs. #265's spike, which did run it) — this spike only measures
  tool selection and argument correctness, not whether the final prose
  answer is faithful to the tool output.
- **#285 interaction**: #285 (sub-agency/office targeting on
  `get_agency_award_breakdown`) touches exactly the same lineage this
  spike's `group_by="sub_agency"` branch would eventually fold in. #285 is
  being implemented independently, directly on the live `get_agency_award_breakdown`,
  since this spike explicitly defers migrating the existing three tools.
  Whenever that migration follow-up happens, it needs to carry #285's
  `sub_agency`/office-targeting params into the consolidated shape, not
  drop them.
- **`district`-style ambiguity risk**: #265's spike found that overloading
  one dimension name across two routes (a category value vs. a `geo_layer`
  value) created real confusion. This spike's two tools don't have an
  analogous overload today, but adding more `group_by` values later (e.g.
  if the four remaining appropriations endpoints from the issue's own
  "next asks" list — `object_class`, `program_activity`, `federal_account`,
  `budget_function` — get folded in) should re-check for this.
- **Real API rate/cost**: the FY2005/FY2004 error-path cases correctly hit
  the live API's actual `fiscal_year < 2008` validation (422) in both arms
  — confirms the harness exercises the real endpoints, not mocks.

## Go/no-go

**Go, with conditions** — proceed toward an implementation PR for the two
new gaps (#286/#287) as `group_by` branches on two consolidated tools,
but treat this as a smaller, lower-stakes decision than #265's:

1. **The token savings alone don't carry the case** (modest -18% vs.
   #265's -84%) — the real argument for consolidating here is avoiding
   the tool-list growth from bolting on two more narrow standalone tools
   (+20% instead), which the issue itself identified as the actual
   problem being solved.
2. **Re-run against the full labeled set through the real `ask()` path**
   (not just this spike's ~20+3 cases via the minimal harness) before
   cutover, same condition #265's spike attached to its own go decision.
3. **Defer migrating the existing three tools** until a follow-up, per
   the issue's own scope — this PR should register only the two new gaps
   as `group_by` branches, not touch `get_agency_budget`/
   `get_agency_award_breakdown`/`get_agency_budget_by_subcomponent`'s
   registration.
4. **Track #285 against the follow-up migration**, not against this PR —
   see the caveat above.
5. **Decide whether `query_agency_award_activity`'s new `group_by=None`
   branch should also replace `get_agency_award_breakdown`'s current
   forced-sub-agency-breakdown behavior for whole-agency questions** even
   before the full three-tool migration — this spike found real
   questions where that's a better answer today, independent of the
   consolidation decision itself.

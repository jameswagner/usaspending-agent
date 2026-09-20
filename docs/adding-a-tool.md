# Adding a Tool

This document walks through the exact files and steps needed to add a new tool to the agent. All five tools that share the common spending filters (`get_spending_by_category`, `get_spending_over_time`, search_awards, etc.) use the exact same registration pattern — if you're adding a new *spending* tool (or modifying the filter list on an existing one), follow this; if you're adding something entirely different (like `lookup_agency`), the steps are the same but the filter list will be empty or minimal.

## The registration flow

### 1. Create the tool module: `backend/app/agent/tools/<name>.py`

If you're unsure what a field means or which of two similarly-named fields
(e.g. `base_exercised_options` vs. `base_and_all_options`) to surface, check
the [Data Dictionary](https://api.usaspending.gov/api/v2/references/data_dictionary/)
before guessing — see CLAUDE.md's "Verify against the live API" section for
why its `Element` names don't match the live JSON keys and how to match by
definition text instead.

Write a `_raw` function (decorated with `@traceable(run_type="tool", name="<name>_raw")`) that takes a `USASpendingClient` and calls the real API. This stays pure and unit-testable — no logging, no tool-call recording, no error presentation. It's where the actual API call happens:

```python
@traceable(run_type="tool", name="my_tool_raw")
def my_tool_raw(
    client: USASpendingClient,
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    # ... other parameters ...
) -> SomeAPIResponse:
    """Internal implementation - call the API, return structured response."""
    # Actual work here
    return client.some_endpoint(...)
```

Then write the public tool function, decorated with `@beta_tool`, that wraps the `_raw` version and handles errors, recordings, and formatting for the model:

```python
@beta_tool
def my_tool(
    # Parameters in the same order as _raw, but with docstrings
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
) -> str:
    """User-facing docstring — what the model sees. First line is a one-liner
    describing when to use this tool. Include all parameters in the docstring's
    Args section.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = my_tool_raw(client, agency_name, start_fiscal_year, end_fiscal_year)
    except USASpendingAPIError as e:
        logger.warning("my_tool failed: %s", e)
        return f"This query failed: {e}."
    
    _record_tool_call("my_tool", response, {"agency_name": agency_name})
    return _wrap_untrusted(format_results(response))
```

### 2. Register in `backend/app/agent/tools/__init__.py`

Add an import statement for both the `_raw` and public function in the module's import block, *and* add both to the `__all__` list:

```python
from .my_tool import my_tool, my_tool_raw

__all__ = [
    # ... existing tools ...
    "my_tool",
    "my_tool_raw",
]
```

The tests `tests/test_langgraph_tools.py::test_langgraph_tools_is_nonempty` and `test_every_tool_schema_matches_its_own_signature` assert that every tool in `__all__` has a corresponding LangChain wrapper in `langgraph_tools.py` with matching signatures.

### 3. Register in `backend/app/agent/langgraph_tools.py`

Import the public tool function and add it to the `_BETA_TOOLS` list:

```python
from .tools import (
    # ... existing imports ...
    my_tool,
)

_BETA_TOOLS = [
    # ... existing tools ...
    my_tool,
]
```

This list is what the LangGraph agent actually sees — the order doesn't matter, but every tool here needs to be in `__all__` above, or the test fails.

### 4. Add a chart/citation decision in `backend/app/agent/response_shaping.py`

Decide whether this tool's results are ever chart-worthy, and add an explicit decision (even if "no, never"). There are two places:

#### In `NEVER_CHART_TOOLS` (if results are never chart-worthy):

```python
NEVER_CHART_TOOLS = {
    "search_guide", "lookup_agency", "search_awards",  # ... existing ...
    "my_tool",  # Add here if results are a single profile, free text, or not data-viz-shaped
}
```

#### In `should_chart()` function (if results *can* be chart-worthy):

```python
if tool_name == "my_tool":
    if len(structured_result) < 2:
        return None  # Single result isn't worth charting
    return ChartSpec(
        chart_type="bar",  # or "line"
        title="My Tool Results",
        labels=[r.name for r in structured_result],
        values=[r.amount for r in structured_result],
    )
```

#### In `build_tool_citation()` function (ALWAYS required):

Even if a tool's results are never charted, it still needs a citation. Add a branch:

```python
if tool_name == "my_tool":
    params = {"agency_name": context["agency_name"]}
    description = f"My Tool: {params['agency_name']}"
    return ToolCitation(
        tool_name=tool_name,
        parameters=params,
        description=description,
        url=...,  # optional
    )
```

Forgetting this step silently returns `None` from `build_tool_citation`, so the tool call has no citation shown to the user — there's no test enforcing it yet (see the issue description), so it only fails if someone notices the missing citation manually. **This is the silent-miss failure mode mentioned in issue #180.**

## Adding or modifying filter parameters

`_build_filters`, `_record_optional_filter_context`, and `_scope_label` (in
`backend/app/agent/tool_filters.py`/`tools/_shared.py`) no longer each declare
their own ~28-parameter signature - all three take `**filters:
Unpack[SpendingFilterParams]` instead, so `SpendingFilterParams` is the single
place that list is defined. `TestSpendingFilterParamsMatchesActualSignatures`
(tests/test_agent.py) fails if any of the three drifts from it, or if
`_build_filters`'s body stops unpacking a field it declares.

If you're adding a new filter (e.g., `new_filter_code`) to a spending tool:

1. Add the field to `SpendingFilterParams` in `backend/app/agent/tool_filters.py`
2. In `_build_filters`: add `new_filter_code = filters.get("new_filter_code")` near
   the top, then the actual validation/transformation logic that turns it into an
   `AdvancedFilters` field - `_record_optional_filter_context` and `_scope_label`
   need **no changes** (they're fully generic over whatever's in `filters`), unless
   the new filter should count as real scope on its own, in which case add it to
   `real_scoping_filters` in `_build_filters` and to `_SCOPE_LABEL_KEYS` in `_shared.py`
3. Add it to every spending tool's `_raw` and `@beta_tool` wrapper functions - this
   part is inherent to exposing a new LLM-facing parameter on each tool and isn't
   eliminated by the TypedDict, since each tool needs its own named parameter for
   `@beta_tool`'s schema generation
4. Add it to `_ALL_OPTIONAL_FILTER_KEYS` in `response_shaping.py`

Steps 3-4 are still manual, per-tool edits - the TypedDict only removed the
duplication across the three *shared* functions, not the tool-specific plumbing.

## Testing

Run the full suite to catch registration misses:

```bash
uv run pytest tests/test_agent.py -xvs
```

The tool-count and schema assertions will fail if a tool is in `__all__` but missing from `_BETA_TOOLS`, or if the `_raw` and `@beta_tool` signatures don't match.

After the chart/citation step, manually test with the live agent to ensure charts render correctly and citations are present — there's no automated test for "citation was shown to the user," so visual inspection is the only gate.

## Add cases to the tool-selection eval

A new tool (or a new filter that makes an existing tool confusable with another) needs at least one entry in `backend/app/agent/dev_tools/tool_selection_labeled_set.json` — this is what actually checks the model *picks* the new tool for the questions it's meant to answer, not just that the tool is wired up correctly. Add:

- A question this tool should clearly be the answer to, with `expected_tool` set to its name.
- If the new tool could plausibly be confused with an existing one (e.g. a new geography/breakdown tool vs. `get_spending_by_category`), also add `confusable_with` naming that tool, and consider a question on the *existing* tool's own entries that this new one might now wrongly steal.

`eval_tool_selection.py` (`backend/app/agent/dev_tools/eval_tool_selection.py`) is what runs this dataset — **do not run it** as part of adding a tool unless the user explicitly asks: it makes real, billed LLM calls against a LangSmith experiment, it's opt-in by design (not part of CI or the regular test suite), and it's the user's call whether that cost is warranted right now. Adding the labeled-set entry is the deliverable; running the eval is a separate, explicit request.

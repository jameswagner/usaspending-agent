# Adding a Tool

This document walks through the exact files and steps needed to add a new tool to the agent. All five tools that share the common spending filters (`get_spending_by_category`, `get_spending_over_time`, search_awards, etc.) use the exact same registration pattern — if you're adding a new *spending* tool (or modifying the filter list on an existing one), follow this; if you're adding something entirely different (like `lookup_agency`), the steps are the same but the filter list will be empty or minimal.

## The registration flow

### 1. Create the tool module: `backend/app/agent/tools/<name>.py`

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

The test `tests/test_agent.py::test_langgraph_tools_match_beta_tools` asserts that every tool in `__all__` has a corresponding LangChain wrapper in `langgraph_tools.py`.

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

If you're adding a new filter (e.g., `new_filter_code`) to a spending tool:

1. Add the parameter to `SpendingFilterParams` in `backend/app/tool_filters.py`
2. Add handling in `_build_filters` (where it becomes an API filter)
3. Add handling in `_record_optional_filter_context` (where it gets recorded for citations)
4. Add handling in `_scope_label` (where it appears in failure/no-results messages)
5. Add it to every spending tool's `_raw` and `@beta_tool` wrapper functions
6. Add it to `_ALL_OPTIONAL_FILTER_KEYS` in `response_shaping.py`

Currently all these edits are manual and easy to miss. The planned `SpendingFilterParams` TypedDict consolidation (see [issue #180](https://github.com/jameswagner/usaspending-agent/issues/180)) will reduce this to ~2 edits once complete.

## Testing

Run the full suite to catch registration misses:

```bash
uv run pytest tests/test_agent.py -xvs
```

The tool-count assertion (`test_langgraph_tools_match_beta_tools`) will fail if a tool is in `__all__` but missing from `_BETA_TOOLS`, or vice versa. The schema-vs-signature check will catch a mismatch between the `_raw` and `@beta_tool` function signatures.

After the chart/citation step, manually test with the live agent to ensure charts render correctly and citations are present — there's no automated test for "citation was shown to the user," so visual inspection is the only gate.

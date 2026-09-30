# Spike #311: download as a normal tool

Spike, not an implementation PR. `backend/app/agent/download_tool_spike.py`
prototypes both shapes without importing them from `langgraph_tools.py` or
`tools/__init__.py`. The live `ask()` download gate and prompt are untouched.
This branch includes the still-unmerged single-award work as a prerequisite;
the spike must be rebased or retargeted when that work lands.

## Two tool shapes prototyped

**Shape A:** `download_data(award_id=None, agency_name=None,
start_fiscal_year=None, end_fiscal_year=None, start_date=None, end_date=None,
award_type=None, spending_level="awards", file_format="csv")`. An ID routes by
`_award_id_endpoint`; without an ID, record scope routes to `/download/search/`.
The function rejects mixed ID and record filters and incomplete or competing
date ranges.

**Shape B:** `download_single_award(award_id)` and
`download_records(agency_name=None, start_fiscal_year=None,
end_fiscal_year=None, start_date=None, end_date=None, award_type=None,
spending_level="awards", file_format="csv")`. The route distinction is in the
tool name.

Both reuse `_build_filters`, the existing columns and spending-level maps,
`_award_id_endpoint`, `_poll_until_finished`, and the existing client methods.
The prototype records a `DownloadSpec` through `_record_tool_call` and checks
the normal per-turn tool budget. It does not alter production registration.

## Token cost (real `bind_tools()` path)

The standalone harness constructs the full production `LANGGRAPH_TOOLS` list,
converts it with `convert_to_anthropic_tool`, marks the last tool and system
block for caching, and binds it with `ChatAnthropic.bind_tools`. Its baseline
retains the system-prompt download carve-out; both prototype arms remove that
text. The one-line tool docstrings follow this repository's comment rule, so
the harness uses `parse_docstring=False` for the three prototype wrappers;
this is a schema conversion difference from production's wrappers.

**Measurement pending.** Automatic approval review rejected the real Anthropic
call because it sends repository prompt and tool schemas using `.env`
credentials. The requested approval has not yet been granted. No token total,
percentage saving, or cold/warm-cache claim is reported from that failed call.

Raw harness output from the attempted run, reduced to the relevant error:

```text
anthropic.APIConnectionError: Connection error.
```

The local sandbox first blocked network resolution. An escalated run was then
rejected before execution by automatic approval review. Consequently there is
no `cache_read=0` observation to establish a cold measurement.

## Accuracy: standalone harness

`spike_311_harness.py` loads the single-turn download cases from
`download_labeled_set.json`, adds two compound questions, and replays the two
specified multi-turn cases. It reuses `tool_selection_correct`,
`tool_order_correct`, and `tool_args_correct` from
`eval_tool_selection.py` on locally constructed trajectories. The same
relevant normal tools are available in all arms. Data-tool replies are stubs,
so these calls would measure model routing and argument carry-over rather than
the accuracy of USAspending results. The baseline has no model-visible
download tool and remains structurally unable to satisfy the download part of
a compound question through this loop. Its current pre-loop handler is not
represented as a tool call in these evaluator functions.

**No billed accuracy run was made** after the automatic approval rejection.
There are no selection percentages to compare. Shape A would make a
name-only `tool_selection_correct` pass for any download route once it calls
`download_data`; its argument score and explicit no-invented-`award_type`
check must carry that distinction. The two compound cases and the grant
follow-up have no valid baseline `download_*` tool, so baseline percentages
would need to be read as a structural comparison, not as a fair measure of
the current handler's clean single-intent download ability.

## Integration findings

**Streaming:** the actual `sse_event_generator` and `_run_graph_stream` path
was exercised with a graph whose tool step blocks for 16 seconds, longer than
the real 15-second keepalive interval. Raw SSE output:

```text
0.00s event: tool_call_start; download_records
15.01s : keep-alive
16.01s event: tool_result; CSV download ready: probe.zip
16.01s event: done; downloads=[]
```

This confirms that the existing SSE bridge emits a keepalive while a graph
tool step blocks. The graph was simulated, so this does not establish a real
90-second download job through a prototype-bound production graph.

**Result projection:** extending `build_tool_citation` alone is insufficient.
The current `_build_result` walks `_tool_call_log` for charts and citations,
but never appends to `AgentResult.downloads`. A direct probe recorded a
`DownloadSpec` under the prototype tool name and returned:

```text
{'downloads': 0, 'citations': 0, 'chart': None, 'direct_citation': None}
```

The SSE `done` frame above likewise contained `downloads=[]`. A follow-up
implementation needs an explicit download projection in `_build_result`, a
download citation branch, and an explicit never-chart decision. It should
test the serialized `/ask/stream` done payload as well as `ask()`.

**Live API:** the public [USAspending endpoint index](https://api.usaspending.gov/docs/endpoints)
lists `/download/search/`, `/download/contract/`, `/download/assistance/`,
`/download/idv/`, and `/download/status/`. A real POST to `/download/contract/`
with `CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-` returned HTTP 200, a job
name and status URL; a status GET returned HTTP 200, `status="finished"` and
`total_rows=71` after `seconds_elapsed="7.947483"`. A POST to
`/download/assistance/` with the grant ID used by the existing unit test
returned HTTP 400, `"Unable to find award matching the provided award id"`;
that ID is suitable for routing tests but not for live end-to-end assertions.
An unknown status filename returned HTTP 404, consistent with the existing
early-404 tolerance. These probes verify the public endpoint behavior used by
the prototype; they do not validate every filter/column combination.

## What this measurement does not establish

- Cold or warm token savings, including the carve-out's actual token cost.
- Relative tool-selection accuracy, compound-question completion, or the two
  multi-turn regressions under billed model calls.
- A real 90-second job through a prototype-bound streaming graph.
- That the model preserves mutually exclusive Shape A arguments or avoids
  inventing `award_type` in a continuation.
- End-to-end citation and download serialization after a production cutover.

The clean full suite (819 tests) exercises existing behavior; it cannot fill
those measurement gaps. The single-award work has not merged into `main` yet,
so this branch also has a prerequisite merge in its history.

## Go/no-go

**No-go for cutover or deleting follow-up machinery yet.** The result
projection needs a concrete code change, and the decisive billed comparison
and multi-turn replays have not run. The streaming keepalive itself looks
viable, but the simulated graph probe is narrower than the requested live
integration test.

Before reconsidering, run the three-arm cold/warm measurement and confirm
`cache_read=0` on each cold arm; run the standalone evaluator and inspect raw
per-case trajectories, including both compounds and both follow-ups; then
verify a real download tool call through streaming with the download and
citation present in the `done` payload. Shape B is the safer provisional
choice because single-award and record downloads are distinct operations,
but the spike has no empirical accuracy basis to prefer it yet.

## Verification

- `uv run pytest tests/ -q`: 819 passed with network access. One existing
  client test reaches the public API despite its fixture; sandbox-only run
  failed on DNS resolution.
- `uv run ruff check` on all added Python files: clean.
- The prototype is not registered in the live tool list.

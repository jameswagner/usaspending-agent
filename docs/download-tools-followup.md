# Download tools follow-up

The two-tool design from the download spike is implemented alongside the existing download handler. The production keyword and follow-up gate remains in place by default. The normal tool loop now has `download_records` and `download_single_award`; set `DOWNLOAD_TOOL_LOOP_ENABLED=1` on the backend to send explicit download requests through those tools for local trials. The flag switches both `/ask` and `/ask/stream` and changes the system prompt to describe the tool path.

## Endpoint and schema decisions

- The [upstream download/search contract](https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/download/search.md) accepts a list of `psc_codes` and a `spending_level` array. A live NSF January 2024 request with `psc_codes=["DA01", "D302"]` and `spending_level=["awards"]` completed with 14 rows in one CSV inside the returned ZIP.
- `download_records` exposes the shared spending filters and accepts several PSCs in one job. Its default is an awards-only CSV; `include_subawards=True` explicitly requests the bundled export. It rejects `recipient_id`, which the download endpoint cannot preserve, instead of generating a broader file.
- `query_spending` still accepts one `psc_code` per call. The wrapper now rejects unknown arguments, so a model-supplied `psc_codes` list produces an error rather than silently falling back to unfiltered spending.
- A recorded download reaches `AgentResult.downloads`, the SSE `done` payload, and a tool citation with the POST request. Download failures produce a `tool_error` event.

## Routing evidence

The [raw 21-case run](evidence/download-tools-full-surface.jsonl) used the full 29-tool schema, billed Anthropic calls, and stubbed tool outputs. It combines 13 cases from one run and the remaining 8 from a second run after the first harness process stopped on an unbound tool name. The harness now records that condition as a failed call. These are model routing measurements, not live spending results.

| Prompt | Observed result |
| --- | --- |
| “Download one CSV of NSF FY2024 contracts with PSC codes DA01 or D302.” | One `download_records` call with both codes; the new default requests `spending_level=["awards"]`. |
| “How much did NSF spend on contracts with PSC DA01 or D302 in FY2024, and download matching awards in one CSV?” | Two matching single-PSC totals and one combined download, but the model added the totals in prose without `sum_values`. |
| “How much did NSF spend on IT contracts in FY2024, and can you also download that as a CSV?” | The amount calls and CSV had different topic and award-type filters, and the model omitted the required arithmetic call. The answer did not establish one matched amount and file. |
| NSF transaction follow-up “How about FY2024?” | Preserved transaction level and changed the fiscal year without inventing an award type. |
| Grant detail follow-up “Download this grant.” | Selected `download_single_award` with the prior full award ID. |
| Prior spending follow-ups with NAICS, geography, program, and recipient ID | Preserved the first three filters; recipient ID reached the explicit unsupported-scope error. |

All 21 cases selected the expected tool family. That score alone misses the two compound failures above. In a separate three-repeat exact-PSC probe with stronger system guidance, the model used one combined download, matched the two PSC scopes, and called `sum_values` in all three repeats. With only basic guidance and the refined descriptions, it selected one combined download in all three repeats but once tried the unsupported `psc_codes` argument on `query_spending`; strict validation caused a visible error and retry. The model also omitted the one-file flag in all three basic runs, which led to the safer tool default.

## Verification and cutover decision

The full repository suite passed with network access: 843 tests. Ruff passed on every touched Python file. A graph/SSE test verified that a download from the normal tool loop reaches the final download and citation payload. The two-PSC, awards-only request was verified against the live public API.

An opt-in local frontend trial used “Download one CSV of NSF contracts from January 2024 with PSC codes DA01 or D302.” The frontend proxy streamed one `download_records` call and a finished 14-row award download with a POST citation. The API's ZIP filename mentions awards, transactions, and subawards even when the request contains only `spending_level=["awards"]`; the tool result now states the requested level so the answer does not infer contents from that generic name. The first trial's wider mixed award-column set failed asynchronously with only “An error occurred”; the new tool uses the four-column contract award set that completed against the live endpoint. Failed jobs now emit `tool_error` and avoid claiming the filters were invalid without evidence.

Keep the legacy gate for now. The compound broad-topic prompt remains unreliable even with a focused tool docstring and system guidance, and the exact-PSC compound prompt still violated the application's arithmetic-tool rule in the full-surface run. A production cutover needs a deterministic scope choice for broad topics and a guard against unsupported or unmatched compound answers. The existing pre-loop handler and follow-up state should remain until those behaviors are validated end to end.

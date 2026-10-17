# Spike #311: download as a normal tool

Spike, not an implementation PR. `backend/app/agent/download_tool_spike.py`
prototypes both shapes without importing them from `langgraph_tools.py` or
`tools/__init__.py`. The live `ask()` keyword gate, follow-up machinery, and
system-prompt carve-out are unchanged. The single-award prerequisite merged
in PR #307 while this spike was in progress; current `main` now contains it.

## Two tool shapes prototyped

**Shape A:** `download_data(award_id=None, agency_name=None,
start_fiscal_year=None, end_fiscal_year=None, start_date=None, end_date=None,
award_type=None, spending_level="awards", keywords=None, psc_code=None,
naics_code=None, file_format="csv")`. An ID routes by
`_award_id_endpoint`; without an ID, record scope routes to `/download/search/`.
The function rejects mixed ID and record filters and incomplete or competing
date ranges.

**Shape B:** `download_single_award(award_id)` and
`download_records(agency_name=None, start_fiscal_year=None,
end_fiscal_year=None, start_date=None, end_date=None, award_type=None,
spending_level="awards", keywords=None, psc_code=None, naics_code=None,
file_format="csv")`. The route distinction is in the
tool name.

Both reuse `_build_filters`, the existing columns and spending-level maps,
`_award_id_endpoint`, `_poll_until_finished`, and the existing client methods.
The record routes expose and forward keyword, PSC, and NAICS filters supported
by `/download/search/`; a filtered contract download was accepted live.
The prototype records a `DownloadSpec` through `_record_tool_call` and checks
the normal per-turn tool budget. It does not alter production registration.

## Token cost (real `bind_tools()` path)

The standalone harness converts the full production `LANGGRAPH_TOOLS` list
with `convert_to_anthropic_tool`, marks the last tool and system block for
one-hour caching, and binds with `ChatAnthropic.bind_tools`. The baseline
retains the system-prompt download carve-out; both prototype arms remove it.
Each arm's first tool description received a same-length random cache nonce,
so shared prefixes could not make a nominally cold arm warm. This adds a small
equal overhead to all arms. The prototype wrappers use `parse_docstring=False`
to honor this repository's one-line docstring rule; production wrappers use
`parse_docstring=True`. That is a schema conversion difference to revisit in
an implementation.

| Arm | Cold input tokens | Cold cache read | Warm cache read | Change vs baseline |
|---|---:|---:|---:|---:|
| Current tools + carve-out | 20,530 | 0 | 20,195 | — |
| Shape A, carve-out removed | 20,926 | 0 | 20,592 | +396 (+1.9%) |
| Shape B, carve-out removed | 20,967 | 0 | 20,633 | +437 (+2.1%) |

Warm requests had the same total logical input tokens and 334–335 uncached
tokens per arm; the difference moved into cached input. The cold observations
really were cold (`cache_read_input_tokens=0` for all three). A separate real
`messages.count_tokens` comparison with identical baseline tools and user
message measured the carve-out at **98 input tokens**: 20,508 with it versus
20,410 without it. Thus, in these exact shapes, the new tool schemas cost more
than removing the carve-out saves. Raw API usage and count output:

```jsonl
{"arm": "baseline", "cache": "cold", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 20195, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 20195, "cache_read_input_tokens": 0, "inference_geo": "not_available", "input_tokens": 335, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20530, "output_tokens": 16, "total_tokens": 20546, "input_token_details": {"cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 20195}}}
{"arm": "baseline", "cache": "warm", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20195, "inference_geo": "not_available", "input_tokens": 335, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20530, "output_tokens": 16, "total_tokens": 20546, "input_token_details": {"cache_read": 20195, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}}
{"arm": "A", "cache": "cold", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 20592, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 20592, "cache_read_input_tokens": 0, "inference_geo": "not_available", "input_tokens": 334, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20926, "output_tokens": 16, "total_tokens": 20942, "input_token_details": {"cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 20592}}}
{"arm": "A", "cache": "warm", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20592, "inference_geo": "not_available", "input_tokens": 334, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20926, "output_tokens": 16, "total_tokens": 20942, "input_token_details": {"cache_read": 20592, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}}
{"arm": "B", "cache": "cold", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 20633, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 20633, "cache_read_input_tokens": 0, "inference_geo": "not_available", "input_tokens": 334, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20967, "output_tokens": 16, "total_tokens": 20983, "input_token_details": {"cache_read": 0, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 20633}}}
{"arm": "B", "cache": "warm", "usage": {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 0}, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 20633, "inference_geo": "not_available", "input_tokens": 334, "output_tokens": 16, "output_tokens_details": null, "server_tool_use": null, "service_tier": "standard"}, "usage_metadata": {"input_tokens": 20967, "output_tokens": 16, "total_tokens": 20983, "input_token_details": {"cache_read": 20633, "cache_creation": 0, "ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0}}}
```

```jsonl
{"carve_out": "with_carve_out", "input_tokens": 20508}
{"carve_out": "without_carve_out", "input_tokens": 20410}
```

## Accuracy: standalone harness

`spike_311_harness.py` loads nine single-turn download cases from
`download_labeled_set.json`, adds two compound questions, and replays the two
specified multi-turn cases. It reuses `tool_selection_correct`,
`tool_order_correct`, and `tool_args_correct` from
`eval_tool_selection.py` on locally constructed trajectories. Each arm sees
the same four relevant normal tools (`query_spending`, `get_award_details`,
`lookup_agency`, `resolve_psc_code`), with the candidate download tools
added only in A and B.
The system instruction is: “Answer federal spending questions with tools.
Create a CSV file only when asked. Preserve prior tool arguments on
follow-ups.” The baseline also receives its production carve-out. Tool
results are stubs, so this measures route choice and arguments, not live
spending results or answer fidelity. One Haiku run per case was made.

| Arm | `tool_selection_correct` | `tool_args_correct` | Effective download level | Compound scope match | Both follow-up routes |
|---|---:|---:|---:|---:|---:|
| Baseline loop | 0/13 | not graded | 0/9 | 0/2 | 0/2 |
| Shape A | 13/13 | 8/8 graded | 9/9 | 2/2 | 2/2 |
| Shape B | 13/13 | 8/8 graded | 9/9 | 2/2 | 2/2 |

The baseline loop's zero is structural: today's download handler runs before
the loop, and has no model-visible download tool. This is **not** a finding
that its existing single-intent downloads fail. The follow-up histories were
synthetic tool messages, not full production conversations. Both candidate
arms carried `spending_level="transactions"` to FY2024 without inventing
`award_type` in the #305 replay, and both passed the prior grant ID to the
assistance route in the #297 replay. The grant ID in the fixture is not a
valid live award; this tests routing only. These successes are insufficient
grounds to delete the production follow-up machinery.

The first version of this spike omitted topic filters from both download
schemas. It therefore could not represent the compound IT question fairly:
both model calls queried a narrow IT scope and downloaded all contracts.
Shape A also invented `award_type="contracts"` for “Can I download NSF's
January 2024 awards as a CSV?” while today's production extractor returned
`award_type=null` in a direct billed comparison. Those are findings about
the **discarded, under-specified prototype**, not the revised shapes. The
current prototypes expose `keywords`, `psc_code`, and `naics_code` in both
record routes and pass them to `_build_filters`.

On the revised run, both shapes omitted `award_type` for the plain-awards
question and the NSF transactions follow-up. The compound IT calls now
carry topic scope to the download tool:

- Shape A resolved PSC candidates but chose `keywords="IT information
  technology"` for both `query_spending` and `download_data`.
- Shape B queried `psc_code="DA01"` and `"D302"` in two spending calls and
  made two corresponding `download_records` calls with the same codes. The
  endpoint accepts PSC filters, but those two codes are **not an exhaustive
  definition of IT contracts**; the resolver stub offered only those two
  candidates. This checks matching scopes between calls, not that the model
  answered all possible meanings of “IT.”

The compound-scope check compares the full multiset of spending and download
calls on agency, fiscal years, award type, topic filter, and record level; it
rejects missing topic filters and invalid-length PSC codes. A first pass of
the harness incorrectly counted three default-`awards` cases as bad
arguments because the model omitted the optional `spending_level` field.
The tool defaults it to `"awards"`. The raw billed trajectories were
rescored offline with the same evaluator functions, grading only non-default
level arguments and separately checking all nine effective levels. No
additional model calls were made for that correction.

Shape A's one download name still weakens name-only selection: that score
cannot distinguish record versus single-award routing. The two compound
cases have no `expected_args` for the stock evaluator, so its argument score
is inapplicable there; the explicit scope check supplies that resolution.
No repeat trials or full production tool surface were used.

Raw case-level harness output follows. Each line includes the actual tool
arguments, so the remaining limits above remain visible alongside the scores:

```jsonl
{"arm": "baseline", "case": "download_0", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "no_invented_award_type": false, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_1", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_2", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_3", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_4", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_5", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_6", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_7", "trajectory": [{"tool": "query_spending", "args": {"agency_name": "National Science Foundation", "level": "subaward", "start_year": 2024, "end_year": 2024, "limit": 5000}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "download_8", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": false}
{"arm": "baseline", "case": "compound_it", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "resolve_psc_code", "args": {"description": "information technology IT services"}, "output": "Candidate PSC codes: DA01 (application development support), D302 (systems development)."}, {"tool": "query_spending", "args": {"level": "award", "agency_name": "National Science Foundation", "award_type": "contracts", "start_year": 2024, "end_year": 2024, "keywords": "IT information technology", "limit": 100}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": 0.0, "tool_args_correct": null}, "compound_scope_match": false}
{"arm": "baseline", "case": "compound_grants", "trajectory": [{"tool": "query_spending", "args": {"level": "award", "award_type": "grants", "agency_name": "HHS", "start_year": 2023, "end_year": 2023, "limit": 5}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": 0.0, "tool_args_correct": null}, "compound_scope_match": false}
{"arm": "baseline", "case": "nsf_followup", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}, "no_invented_award_type": false}
{"arm": "baseline", "case": "grant_followup", "trajectory": [], "scores": {"tool_selection_correct": 0.0, "tool_order_correct": null, "tool_args_correct": null}}
{"arm": "A", "case": "download_0", "trajectory": [{"tool": "download_data", "args": {"agency_name": "National Science Foundation", "start_date": "2024-01-01", "end_date": "2024-01-31", "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "no_invented_award_type": true, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_1", "trajectory": [{"tool": "download_data", "args": {"agency_name": "Health and Human Services", "award_type": "grants", "start_fiscal_year": 2023, "end_fiscal_year": 2023, "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_2", "trajectory": [{"tool": "lookup_agency", "args": {"name": "Department of Education"}, "output": "Tool returned successfully."}, {"tool": "download_data", "args": {"agency_name": "Department of Education", "award_type": "contracts", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_3", "trajectory": [{"tool": "download_data", "args": {"agency_name": "National Science Foundation", "spending_level": "transactions", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_4", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "download_data", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_5", "trajectory": [{"tool": "download_data", "args": {"agency_name": "National Science Foundation", "spending_level": "subawards", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_6", "trajectory": [{"tool": "download_data", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "subawards"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_7", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "download_data", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "subawards"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "download_8", "trajectory": [{"tool": "download_data", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions", "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "A", "case": "compound_it", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "resolve_psc_code", "args": {"description": "IT contracts information technology"}, "output": "Candidate PSC codes: DA01 (application development support), D302 (systems development)."}, {"tool": "query_spending", "args": {"level": "award", "agency_name": "National Science Foundation", "award_type": "contracts", "start_year": 2024, "end_year": 2024, "keywords": "IT information technology", "limit": 500}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}, {"tool": "download_data", "args": {"agency_name": "National Science Foundation", "award_type": "contracts", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "keywords": "IT information technology", "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": 1.0, "tool_args_correct": null}, "compound_scope_match": true}
{"arm": "A", "case": "compound_grants", "trajectory": [{"tool": "query_spending", "args": {"level": "award", "award_type": "grants", "agency_name": "HHS", "start_year": 2023, "end_year": 2023, "limit": 5}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}, {"tool": "download_data", "args": {"award_type": "grants", "agency_name": "HHS", "start_fiscal_year": 2023, "end_fiscal_year": 2023}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": 1.0, "tool_args_correct": null}, "compound_scope_match": true}
{"arm": "A", "case": "nsf_followup", "trajectory": [{"tool": "download_data", "args": {"agency_name": "NSF", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "no_invented_award_type": true}
{"arm": "A", "case": "grant_followup", "trajectory": [{"tool": "download_data", "args": {"award_id": "ASST_NON_H79TI081692_7522"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}}
{"arm": "B", "case": "download_0", "trajectory": [{"tool": "download_records", "args": {"agency_name": "National Science Foundation", "start_date": "2024-01-01", "end_date": "2024-01-31", "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "no_invented_award_type": true, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_1", "trajectory": [{"tool": "download_records", "args": {"agency_name": "Health and Human Services", "award_type": "grants", "start_fiscal_year": 2023, "end_fiscal_year": 2023}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_2", "trajectory": [{"tool": "download_records", "args": {"agency_name": "Department of Education", "award_type": "contracts", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": null}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_3", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "download_records", "args": {"agency_name": "National Science Foundation", "spending_level": "transactions", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_4", "trajectory": [{"tool": "download_records", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_5", "trajectory": [{"tool": "download_records", "args": {"agency_name": "National Science Foundation", "spending_level": "subawards", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_6", "trajectory": [{"tool": "download_records", "args": {"agency_name": "National Science Foundation", "award_type": "grants", "spending_level": "subawards", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "file_format": "csv"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_7", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "download_records", "args": {"agency_name": "National Science Foundation", "award_type": "grants", "spending_level": "subawards", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "download_8", "trajectory": [{"tool": "download_records", "args": {"agency_name": "National Science Foundation", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "effective_spending_level_correct": true}
{"arm": "B", "case": "compound_it", "trajectory": [{"tool": "lookup_agency", "args": {"name": "National Science Foundation"}, "output": "Tool returned successfully."}, {"tool": "resolve_psc_code", "args": {"description": "information technology"}, "output": "Candidate PSC codes: DA01 (application development support), D302 (systems development)."}, {"tool": "query_spending", "args": {"level": "award", "award_type": "contracts", "agency_name": "National Science Foundation", "psc_code": "DA01", "start_year": 2024, "end_year": 2024, "limit": 5}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}, {"tool": "query_spending", "args": {"level": "award", "award_type": "contracts", "agency_name": "National Science Foundation", "psc_code": "D302", "start_year": 2024, "end_year": 2024, "limit": 5}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}, {"tool": "download_records", "args": {"agency_name": "National Science Foundation", "award_type": "contracts", "psc_code": "DA01", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}, {"tool": "download_records", "args": {"agency_name": "National Science Foundation", "award_type": "contracts", "psc_code": "D302", "start_fiscal_year": 2024, "end_fiscal_year": 2024}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": 1.0, "tool_args_correct": null}, "compound_scope_match": true}
{"arm": "B", "case": "compound_grants", "trajectory": [{"tool": "query_spending", "args": {"level": "award", "agency_name": "Department of Health and Human Services", "award_type": "grants", "start_year": 2023, "end_year": 2023}, "output": "NSF IT contracts FY2024 total: $12,345,678; records available for export."}, {"tool": "download_records", "args": {"spending_level": "awards", "award_type": "grants", "agency_name": "Department of Health and Human Services", "start_fiscal_year": 2023, "end_fiscal_year": 2023}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": 1.0, "tool_args_correct": null}, "compound_scope_match": true}
{"arm": "B", "case": "nsf_followup", "trajectory": [{"tool": "download_records", "args": {"agency_name": "NSF", "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}, "no_invented_award_type": true}
{"arm": "B", "case": "grant_followup", "trajectory": [{"tool": "download_single_award", "args": {"award_id": "ASST_NON_H79TI081692_7522"}, "output": "CSV download ready: example.csv."}], "scores": {"tool_selection_correct": 1.0, "tool_order_correct": null, "tool_args_correct": 1.0}}
```

## Endpoint scope and integration findings

The upstream [`/download/search/` contract](https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/contracts/v2/download/search.md)
accepts an `AdvancedFilters`-style `filters` object with `keywords`,
`psc_codes`, `naics_codes`, agency, time, award type, recipient, location,
award ID, and other filters. It also accepts `columns`, `limit`,
`spending_level`, and `file_format` (`csv`, `tsv`, `pstxt`). The upstream
[shared search-filter contract](https://github.com/fedspendingtransparency/usaspending-api/blob/master/usaspending_api/api_contracts/search_filters.md)
allows filter fields to combine. Our `_build_filters` already converts
`keywords`, validated `psc_code`, and `naics_code` inputs to the corresponding
API fields; the revised prototypes forward all three. A real
`/download/search/` POST with NSF, January 2024, contracts, and
`keywords=["information technology"]` was accepted and returned a job. A
coherent “amount and matching CSV” workflow must choose what “IT” means,
then apply the same topic scope, agency, dates, award type, and spending
level to both operations. A spending amount and award-row CSV can still
differ by accounting basis; matching filters alone cannot establish that
the CSV's rows sum to the reported figure.

**Streaming:** `spike_311_stream_probe.py` binds the actual prototype
`download_records` tool into a standalone `create_react_agent` graph and
drives the real `sse_event_generator`/`_run_graph_stream` path. A simulated
16-second status wait produced a keepalive at 15 seconds. A live NSF January
2024 `/download/search/` job then occupied the tool call for 95.92 seconds;
SSE emitted keepalives at approximately 15, 30, 45, 60, 75, and 90 seconds.
The job was still generating at the existing polling timeout, and the tool
returned the expected “still generating” message. A scripted model supplied
the final prose, so its “ready” text is not a model-quality observation.
Raw live SSE output:

```jsonl
{"elapsed_seconds": 0.0, "frame": "event: tool_call_start\ndata: {\"tool_name\": \"download_records\", \"args\": {\"agency_name\": \"NSF\", \"start_date\": \"2024-01-01\", \"end_date\": \"2024-01-31\"}}"}
{"elapsed_seconds": 15.01, "frame": ": keep-alive"}
{"elapsed_seconds": 30.01, "frame": ": keep-alive"}
{"elapsed_seconds": 45.01, "frame": ": keep-alive"}
{"elapsed_seconds": 60.02, "frame": ": keep-alive"}
{"elapsed_seconds": 75.02, "frame": ": keep-alive"}
{"elapsed_seconds": 90.02, "frame": ": keep-alive"}
{"elapsed_seconds": 95.92, "frame": "event: tool_result\ndata: {\"tool_name\": \"download_records\", \"summary\": \"CSV download still generating: PrimeAwardsTransactionsAndSubawards_2026-09-30_H19M01S35028830.zip. Status: https://api.usaspending.gov/api/v2/download/status?file_name=PrimeAwardsTransactionsAndSubawa\\u2026\"}"}
{"elapsed_seconds": 95.92, "frame": "event: done\ndata: {\"answer_text\": \"The CSV is ready.\", \"source_type\": \"agent\", \"conversation_id\": \"probe\", \"charts\": [], \"citations\": [], \"tool_citations\": [], \"downloads\": []}"}
```

**Result projection:** extending `build_tool_citation` alone is insufficient.
The current `_build_result` walks `_tool_call_log` for charts and citations,
but never appends to `AgentResult.downloads`. A direct probe recorded a
`DownloadSpec` under the prototype tool name and returned:

```text
{'downloads': 0, 'citations': 0, 'chart': None, 'direct_citation': None}
```

The live SSE `done` frame likewise contained `downloads=[]`. A follow-up
implementation needs an explicit download projection in `_build_result`, a
download citation branch, and an explicit never-chart decision. It should
test the serialized `/ask/stream` done payload as well as `ask()`.

**Other live API checks:** the public
[endpoint index](https://api.usaspending.gov/docs/endpoints) lists the three
single-award endpoints, `/download/search/`, and `/download/status/`. A POST
to `/download/contract/` with
`CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-` returned HTTP 200; its status
GET returned `finished` and 71 rows after 7.95 seconds. The fixture grant
ID returned HTTP 400 (`Unable to find award matching the provided award id`)
from `/download/assistance/`. An unknown status filename returned HTTP 404,
consistent with the existing early-404 tolerance.

## What this measurement does not establish

- A production `ask()` cutover result: the prototype was never registered in
  the live tool list, and the selection harness used only four normal tools.
- Repeat-trial error rates, answer correctness, or a valid IT spending total.
- That two PSC codes exhaust the ordinary meaning of “IT contracts,” or that
  a keyword search and a PSC search identify equivalent awards.
- That the model preserves every filter supported by `/download/search/`;
  the prototype exposes only the fields needed for these cases.
- Full production multi-turn behavior after deleting the custom follow-up
  machinery; the two replay histories were scripted.
- End-to-end citation and download serialization after an implementation.

## Go/no-go

**Go for a follow-up implementation experiment; no-go for immediate cutover
or deletion of follow-up machinery.** The revised shapes passed this narrow
selection, scope-matching, and two-follow-up replay set, and the real
96-second streaming path stayed alive. This supports moving download into
the tool loop as a concrete implementation target. It does not prove the
full production behavior: both shapes add roughly 2% prompt tokens despite
removing the carve-out, the model's definition of broad “IT” remains
under-specified, and `_build_result` currently drops the download from
`AgentResult`.

Shape B is a tentative starting point, since a single-award export and a
filtered record export remain separate named operations for the model, at
only 41 more tokens than Shape A. The 13-case run does not empirically
separate their selection accuracy. The implementation should keep relevant
`_build_filters` fields available on the record tool, add explicit download
projection/citation handling, then rerun against the fuller production tool
surface and real API results. Verify equivalent filters and accounting
basis on compound questions, negative argument assertions, two real
multi-turn histories, and a populated download/citation in the SSE done
payload before replacing the handler and deleting its follow-up code.

## Verification

- `uv run pytest tests/ -q`: 819 passed with network access after the revised
  prototype. In the sandbox, the one existing network-dependent client test
  failed on DNS resolution; the other 818 passed.
- `uv run ruff check` on all added Python files: clean.
- The prototype remains absent from the production tool list.

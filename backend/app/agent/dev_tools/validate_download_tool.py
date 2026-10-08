"""Billed routing probe with the complete production tool schema."""
from __future__ import annotations

import json
import os
from collections import Counter
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from backend.app.agent.dev_tools.eval_download_handler import load_labeled_set
from backend.app.agent.dev_tools.eval_tool_selection import (
    tool_args_correct,
    tool_order_correct,
    tool_selection_correct,
)
from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS
from backend.app.agent.orchestrator import _build_system_prompt
from backend.app.agent.singletons import CACHE_TTL, MODEL

CARVE_OUT = (
    "Separately from these tools, a CSV download of an agency's awards "
    "IS available - it's handled automatically outside this tool loop "
    "when a question explicitly asks to download/export the data, so "
    "never claim this app can't produce a CSV; if you're here for a "
    "download request, ask the user to rephrase explicitly (e.g. "
    "'download NSF's FY2024 awards as a CSV') rather than pointing them to usaspending.gov."
)
BASIC_GUIDANCE = (
    "Use download_single_award for one known full internal award ID and download_records for "
    "filtered CSV records. For an amount plus a matching CSV, use a spending query and "
    "download_records with identical agency, period, award type, topic, and other scope filters. "
    "A top-N list is not a grand total. For a follow-up download, reuse the prior tool call's "
    "arguments without inventing an award type or silently dropping scope. The download endpoint "
    "cannot filter by recipient ID. When a prior query used recipient_id, pass it to download_records "
    "so the tool returns its explicit scope error; report that limitation without offering a broader file. "
    "A generating job is not a ready file."
    " A named calendar month such as January 2024 means that month's first through last day; "
    "do not ask if the user meant a fiscal year. Omitted award type means all award types."
)
TOPIC_GUIDANCE = (
    " For a broad topic such as IT, use one keyword scope for the amount and one matching CSV; "
    "a few candidate PSC codes do not define all IT contracts. If separate scoped totals must "
    "be combined, call sum_values rather than adding them in prose. A request for a CSV means "
    "one file unless the user asks for several."
)
DOWNLOAD_GUIDANCE = BASIC_GUIDANCE + TOPIC_GUIDANCE
PSC_LIST_GUIDANCE = BASIC_GUIDANCE + (
    " When the user names several PSC codes, use one download_records call with psc_codes as a list "
    "and include_subawards=false for one award CSV. query_spending accepts one psc_code at a time; "
    "query each requested code with group_by=time, call sum_values on those totals, and explain that "
    "the combined amount covers those codes rather than every possible IT contract."
)


def _prior_spending(args: dict) -> list:
    return [
        HumanMessage(content="Show this spending scope."),
        AIMessage(content="", tool_calls=[{"id": "prior-spending", "name": "query_spending", "args": args}]),
        ToolMessage(content="Scoped spending total $12,345,678.", tool_call_id="prior-spending"),
        AIMessage(content="Here is the scoped spending total."),
    ]


def _cases() -> list[tuple[str, str, dict, list]]:
    cases = [
        (f"download_{i}", item["question"], {
            "expected_tool": "download_records",
            **({"expected_args": {"download_records": {
                "spending_level": item["expect_spending_level"],
            }}} if item["expect_spending_level"] != "awards" else {}),
        }, [])
        for i, item in enumerate(load_labeled_set())
        if item.get("expect_download") and "question" in item
    ]
    cases.extend([
        ("amount_it", "How much did NSF spend on IT contracts in FY2024?",
         {"expected_tool": "query_spending"}, []),
        ("download_it", "Download NSF's IT contracts in FY2024 as one CSV.",
         {"expected_tool": "download_records"}, []),
        ("download_two_psc", "Download one CSV of NSF FY2024 contracts with PSC codes DA01 or D302.",
         {"expected_tool": "download_records"}, []),
        ("compound_it", "How much did NSF spend on IT contracts in FY2024, and can you also download that as a CSV?",
         {"expected_tools": ["query_spending", "download_records"]}, []),
        ("compound_two_psc", "How much did NSF spend on contracts with PSC DA01 or D302 in FY2024, and download matching awards in one CSV?",
         {"expected_tools": ["query_spending", "download_records"]}, []),
        ("compound_grants", "How much did HHS spend on grants in FY2023, and download matching awards as CSV?",
         {"expected_tools": ["query_spending", "download_records"]}, []),
        ("nsf_followup", "How about FY2024?",
         {"expected_tool": "download_records", "expected_args": {"download_records": {
             "start_fiscal_year": 2024, "end_fiscal_year": 2024, "spending_level": "transactions",
         }}}, [
             HumanMessage(content="Can I download NSF's transactions from FY2023 as a CSV?"),
             AIMessage(content="", tool_calls=[{"id": "prior-download", "name": "download_records",
                                                "args": {"agency_name": "NSF", "start_fiscal_year": 2023,
                                                         "end_fiscal_year": 2023, "spending_level": "transactions"}}]),
             ToolMessage(content="CSV download ready: nsf-2023.csv.", tool_call_id="prior-download"),
             AIMessage(content="Your NSF FY2023 transaction CSV is ready."),
         ]),
        ("grant_followup", "Download this grant.",
         {"expected_tool": "download_single_award", "expected_args": {"download_single_award": {
             "award_id": "ASST_NON_H79TI081692_7522",
         }}}, [
             HumanMessage(content="Show details for grant ASST_NON_H79TI081692_7522."),
             AIMessage(content="", tool_calls=[{"id": "prior-detail", "name": "get_award_details",
                                                "args": {"award_id": "ASST_NON_H79TI081692_7522"}}]),
             ToolMessage(content="Grant details [internal_id: ASST_NON_H79TI081692_7522]",
                         tool_call_id="prior-detail"),
             AIMessage(content="Here are the grant details."),
         ]),
    ])
    cases.extend([
        ("naics_followup", "Can I download that as a CSV?", {
            "expected_tool": "download_records",
            "expected_args": {"download_records": {"naics_code": "518210"}},
        }, _prior_spending({
            "level": "award", "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "naics_code": "518210", "group_by": "time",
        })),
        ("geo_followup", "Download that as a CSV.", {
            "expected_tool": "download_records",
            "expected_args": {"download_records": {"performed_in_state": "VA"}},
        }, _prior_spending({
            "level": "award", "agency_name": "Department of Defense", "start_year": 2024,
            "end_year": 2024, "performed_in_state": "VA", "group_by": "time",
        })),
        ("program_followup", "Give me a CSV of that same set.", {
            "expected_tool": "download_records",
            "expected_args": {"download_records": {"cfda_program": "47.076"}},
        }, _prior_spending({
            "level": "award", "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "cfda_program": "47.076", "group_by": "time",
        })),
        ("recipient_id_followup", "Download that exact recipient ID scope as CSV.", {
            "expected_tool": "download_records",
            "expected_args": {"download_records": {"recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"}},
        }, _prior_spending({
            "level": "award", "recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", "start_year": 2024,
            "end_year": 2024, "group_by": "time",
        })),
    ])
    return cases


def _output(name: str, args: dict) -> str:
    if name == "resolve_psc_code":
        return "Candidate PSC codes: DA01 (application development support), D302 (systems development)."
    if name == "query_spending":
        if args.get("group_by") == "time":
            return f"Scoped spending result for {args.get('agency_name', 'all agencies')}: aggregate total $12,345,678."
        return "Ranked results only; no grand total in this response."
    if name == "sum_values":
        return "Sum of the supplied values: $24,691,356."
    if name.startswith("download_"):
        if args.get("recipient_id"):
            return "This download failed: the download endpoint cannot filter by recipient ID."
        return "CSV download ready: example.csv, 12 rows."
    if name == "get_award_details":
        return f"Award details [internal_id: {args.get('award_id')}]."
    return "Tool returned successfully."


def _scope_match(name: str, trajectory: list[dict]) -> bool:
    scope_keys = (
        "award_type", "recipient_name", "performed_in_state", "recipient_in_state",
        "keywords", "psc_code", "naics_code", "cfda_program",
    )

    def signature(args: dict, spending: bool) -> tuple | None:
        level = args.get("level") if spending else args.get("spending_level", "awards")
        if level != ("award" if spending else "awards"):
            return None
        agency = args.get("agency_name", "").lower()
        agency = {"nsf": "national science foundation",
                  "hhs": "department of health and human services"}.get(agency, agency)
        topic = tuple((key, args[key]) for key in scope_keys if args.get(key) is not None)
        if name == "compound_it" and not any(key in {"keywords", "psc_code", "naics_code"} for key, _ in topic):
            return None
        return (
            agency, args.get("start_year" if spending else "start_fiscal_year"),
            args.get("end_year" if spending else "end_fiscal_year"), topic,
        )

    totals_args = [
        step["args"] for step in trajectory
        if step["tool"] == "query_spending" and step["args"].get("group_by") == "time"
        and not step.get("invalid_args")
    ]
    downloads_args = [step["args"] for step in trajectory if step["tool"] == "download_records"]
    if len(downloads_args) == 1 and downloads_args[0].get("psc_codes"):
        codes = set(downloads_args[0]["psc_codes"])
        if len(totals_args) != len(codes) or {args.get("psc_code") for args in totals_args} != codes:
            return False
        for args in totals_args:
            combined = dict(downloads_args[0], psc_code=args["psc_code"])
            combined.pop("psc_codes")
            if signature(args, True) != signature(combined, False):
                return False
        return True
    totals = [signature(args, True) for args in totals_args]
    downloads = [signature(args, False) for args in downloads_args]
    return bool(totals and downloads) and None not in totals + downloads and Counter(totals) == Counter(downloads)


def main() -> None:
    load_dotenv(os.environ.get("USSPENDING_ENV_FILE", ".env"))
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
    schemas = [convert_to_anthropic_tool(tool) for tool in LANGGRAPH_TOOLS]
    wrapped_by_name = {tool.name: tool for tool in LANGGRAPH_TOOLS}
    schemas[-1] = dict(schemas[-1], cache_control={"type": "ephemeral", "ttl": CACHE_TTL})
    model = ChatAnthropic(
        model=os.environ.get("AGENT_MODEL", MODEL), max_tokens=1024, default_headers=headers,
    ).bind_tools(schemas)
    prompt = _build_system_prompt()
    assert CARVE_OUT in prompt
    guidance = {
        "basic": BASIC_GUIDANCE,
        "psc_list": PSC_LIST_GUIDANCE,
    }.get(os.environ.get("GUIDANCE"), DOWNLOAD_GUIDANCE)
    prompt = prompt.replace(CARVE_OUT, guidance).replace(
        "have twenty-three tools. Twenty-two retrieve data:", "have tools for data retrieval and downloads:"
    ).replace("twenty-two data tools", "data or download tools")
    requested = set(filter(None, os.environ.get("CASE_NAMES", "").split(",")))
    selected = [case for case in _cases() if not requested or case[0] in requested]
    for index, (name, question, expected, history) in enumerate(selected * int(os.environ.get("REPEATS", "1"))):
        trial = index // len(selected) + 1
        messages = [
            SystemMessage(content=[{"type": "text", "text": prompt,
                                    "cache_control": {"type": "ephemeral", "ttl": CACHE_TTL}}]),
            *history,
            HumanMessage(content=question),
        ]
        trajectory = []
        for _ in range(7):
            reply = model.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                wrapped = wrapped_by_name.get(call["name"])
                if wrapped is None:
                    invalid_args = []
                    output = "This query failed: unknown tool name. Use a tool in the bound schema."
                else:
                    invalid_args = sorted(set(call["args"]) - set(wrapped.args_schema.model_fields))
                    try:
                        wrapped.args_schema.model_validate(call["args"])
                    except ValidationError as error:
                        if callable(wrapped.handle_validation_error):
                            output = wrapped.handle_validation_error(error)
                        else:
                            output = "This query failed: invalid tool arguments."
                    else:
                        output = _output(call["name"], call["args"])
                trajectory.append({
                    "tool": call["name"], "args": call["args"], "output": output,
                    "invalid_args": invalid_args, "unknown_tool": wrapped is None,
                })
                messages.append(ToolMessage(content=output, tool_call_id=call["id"]))
        run = SimpleNamespace(outputs={"tools_called": [step["tool"] for step in trajectory],
                                       "trajectory": trajectory})
        example = SimpleNamespace(outputs=expected)
        scores = {fn.__name__: fn(run, example)["score"] for fn in (
            tool_selection_correct, tool_order_correct, tool_args_correct,
        )}
        extra = {"scope_match": _scope_match(name, trajectory)} if name.startswith("compound_") else {}
        if name in {"amount_it", "download_it", "download_two_psc"}:
            scope_calls = [
                step for step in trajectory
                if step["tool"] == ("query_spending" if name == "amount_it" else "download_records")
                and not step.get("invalid_args")
            ]
            extra["topic_scoped"] = bool(scope_calls) and all(
                any(step["args"].get(key) for key in ("keywords", "psc_code", "psc_codes", "naics_code"))
                for step in scope_calls
            )
            if name in {"download_it", "download_two_psc"}:
                extra["one_file"] = (
                    len(scope_calls) == 1 and scope_calls[0]["args"].get("include_subawards", False) is False
                )
            if name == "download_two_psc":
                extra["both_codes"] = (
                    len(scope_calls) == 1 and set(scope_calls[0]["args"].get("psc_codes", [])) == {"DA01", "D302"}
                )
        if name in {"compound_it", "compound_two_psc"}:
            downloads = [step for step in trajectory if step["tool"] == "download_records"]
            totals = [step for step in trajectory if step["tool"] == "query_spending"
                      and step["args"].get("group_by") == "time" and not step.get("invalid_args")]
            extra["one_file"] = (
                len(downloads) == 1 and downloads[0]["args"].get("include_subawards", False) is False
            )
            extra["arithmetic_ok"] = len(totals) <= 1 or any(
                step["tool"] == "sum_values" for step in trajectory
            )
            if name == "compound_two_psc":
                extra["both_codes"] = (
                    len(downloads) == 1 and set(downloads[0]["args"].get("psc_codes", [])) == {"DA01", "D302"}
                )
        if name == "recipient_id_followup":
            downloads = [step for step in trajectory if step["tool"] == "download_records"]
            extra["safe_recipient_scope"] = (
                all(step["args"].get("recipient_id") for step in downloads)
                and "recipient id" in str(reply.content).lower()
            )
        print(json.dumps({"case": name, "trial": trial, "trajectory": trajectory, "scores": scores,
                          "answer": reply.content, **extra}), flush=True)


if __name__ == "__main__":
    main()

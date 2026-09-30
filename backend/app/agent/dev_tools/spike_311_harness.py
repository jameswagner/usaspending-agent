"""Standalone billed measurement of unwired download tool shapes."""
from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_anthropic.chat_models import convert_to_anthropic_tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from backend.app.agent.dev_tools.eval_download_handler import (
    load_labeled_set as load_download_cases,
)
from backend.app.agent.dev_tools.eval_tool_selection import (
    tool_args_correct,
    tool_order_correct,
    tool_selection_correct,
)
from backend.app.agent.download_tool_spike import (
    download_data,
    download_records,
    download_single_award,
)
from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS
from backend.app.agent.orchestrator import _build_system_prompt
from backend.app.agent.singletons import MODEL

load_dotenv()

CARVE_OUT = (
    "Separately from these tools, a CSV download of an agency's awards "
    "IS available - it's handled automatically outside this tool loop "
    "when a question explicitly asks to download/export the data, so "
    "never claim this app can't produce a CSV; if you're here for a "
    "download request, ask the user to rephrase explicitly (e.g. "
    "'download NSF's FY2024 awards as a CSV') rather than pointing them to usaspending.gov."
)

SHAPES = {
    "baseline": [],
    "A": [download_data],
    "B": [download_single_award, download_records],
}


def _lc_prototypes(shape: str) -> list:
    return [lc_tool(item.func, parse_docstring=False) for item in SHAPES[shape]]


def measure_tokens() -> None:
    for arm in SHAPES:
        schemas = [convert_to_anthropic_tool(tool) for tool in LANGGRAPH_TOOLS + _lc_prototypes(arm)]
        schemas[-1] = dict(schemas[-1], cache_control={"type": "ephemeral", "ttl": "5m"})
        prompt = _build_system_prompt()
        if arm != "baseline":
            prompt = prompt.replace(CARVE_OUT, "")
        prompt += f"\nMeasurement arm: {arm}."
        model = ChatAnthropic(model=MODEL, max_tokens=16).bind_tools(schemas)
        messages = [
            SystemMessage(content=[{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral", "ttl": "5m"}}]),
            HumanMessage(content="How much did NSF spend in FY2024?"),
        ]
        for temperature in ("cold", "warm"):
            response = model.invoke(messages)
            print(json.dumps({"arm": arm, "cache": temperature, "usage": response.response_metadata.get("usage"),
                              "usage_metadata": response.usage_metadata}, default=str))


@dataclass
class Case:
    name: str
    question: str
    expected: dict
    history: list | None = None


CASES = [
    Case(
        f"download_{index}", entry["question"],
        {"expected_tool": "download_records", "expected_args": {"download_records": {"spending_level": entry["expect_spending_level"]}}},
    )
    for index, entry in enumerate(load_download_cases())
    if entry.get("expect_download") and "question" in entry
] + [
    Case("compound_it", "How much did NSF spend on IT contracts in FY2024, and can you also download that as a CSV?", {"expected_tools": ["query_spending", "download_records"]}),
    Case("compound_grants", "Show HHS grant spending in FY2023 and download the records as CSV.", {"expected_tools": ["query_spending", "download_records"]}),
    Case("nsf_followup", "How about FY2024?", {"expected_tool": "download_records", "expected_args": {"download_records": {"spending_level": "transactions", "start_fiscal_year": 2024, "end_fiscal_year": 2024}}}),
    Case("grant_followup", "Download this grant.", {"expected_tool": "download_single_award", "expected_args": {"download_single_award": {"award_id": "ASST_NON_H79TI081692_7522"}}}),
]


def _history(case: Case, arm: str) -> list:
    if case.name == "nsf_followup":
        if arm == "baseline":
            return [
                HumanMessage(content="Can I download NSF's transactions from FY2023 as a CSV?"),
                AIMessage(content="Your NSF FY2023 transaction CSV is ready."),
            ]
        return [
            HumanMessage(content="Can I download NSF's transactions from FY2023 as a CSV?"),
            AIMessage(content="", tool_calls=[{"id": "previous-download", "name": "download_data" if arm == "A" else "download_records", "args": {"agency_name": "NSF", "start_fiscal_year": 2023, "end_fiscal_year": 2023, "spending_level": "transactions"}}]),
            ToolMessage(content="CSV download ready: nsf-2023.csv.", tool_call_id="previous-download"),
            AIMessage(content="Your NSF FY2023 transaction CSV is ready."),
        ]
    if case.name == "grant_followup":
        return [
            HumanMessage(content="Show details for grant ASST_NON_H79TI081692_7522."),
            AIMessage(content="", tool_calls=[{"id": "previous-detail", "name": "get_award_details", "args": {"award_id": "ASST_NON_H79TI081692_7522"}}]),
            ToolMessage(content="Grant details [internal_id: ASST_NON_H79TI081692_7522]", tool_call_id="previous-detail"),
            AIMessage(content="Here are the grant details."),
        ]
    return []


def _tool_output(name: str) -> str:
    if name == "query_spending":
        return "NSF IT contracts FY2024 total: $12,345,678; records available for export."
    if name.startswith("download_"):
        return "CSV download ready: example.csv."
    return "Tool returned successfully."


def evaluate_selection() -> None:
    relevant = [t for t in LANGGRAPH_TOOLS if t.name in {"query_spending", "get_award_details", "lookup_agency"}]
    for arm in SHAPES:
        tools = relevant + _lc_prototypes(arm)
        schemas = [convert_to_anthropic_tool(t) for t in tools]
        model = ChatAnthropic(model=MODEL, max_tokens=500).bind_tools(schemas)
        for case in CASES:
            prompt = "Answer federal spending questions with tools. Create a CSV file only when asked. Preserve prior tool arguments on follow-ups."
            if arm == "baseline":
                prompt += " " + CARVE_OUT
            messages = [SystemMessage(content=prompt), *_history(case, arm), HumanMessage(content=case.question)]
            trajectory = []
            for _ in range(3):
                reply = model.invoke(messages)
                messages.append(reply)
                if not reply.tool_calls:
                    break
                for call in reply.tool_calls:
                    trajectory.append({"tool": call["name"], "args": call["args"], "output": _tool_output(call["name"])})
                    messages.append(ToolMessage(content=_tool_output(call["name"]), tool_call_id=call["id"]))
            expected = json.loads(json.dumps(case.expected))
            if arm == "A":
                for key in ("expected_tool",):
                    if key in expected and expected[key].startswith("download_"):
                        expected[key] = "download_data"
                if "expected_tools" in expected:
                    expected["expected_tools"] = ["download_data" if t.startswith("download_") else t for t in expected["expected_tools"]]
                if "expected_args" in expected:
                    expected["expected_args"] = {"download_data": next(iter(expected["expected_args"].values()))}
            run = SimpleNamespace(outputs={"tools_called": [s["tool"] for s in trajectory], "trajectory": trajectory})
            example = SimpleNamespace(outputs=expected)
            scores = {fn.__name__: fn(run, example)["score"] for fn in (tool_selection_correct, tool_order_correct, tool_args_correct)}
            extra = {}
            if case.name == "nsf_followup":
                calls = [s["args"] for s in trajectory if s["tool"] in {"download_data", "download_records"}]
                extra["no_invented_award_type"] = bool(calls) and all(not args.get("award_type") for args in calls)
            print(json.dumps({"arm": arm, "case": case.name, "trajectory": trajectory, "scores": scores, **extra}, default=str))


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["tokens"]:
        measure_tokens()
    else:
        evaluate_selection()

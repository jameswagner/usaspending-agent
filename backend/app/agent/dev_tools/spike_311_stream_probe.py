"""Exercise SSE framing around a blocking prototype tool in a real graph."""
from __future__ import annotations

import asyncio
import json
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import tool as lc_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from backend.app.agent.download_tool_spike import download_records
from backend.app.agent.streaming import sse_event_generator


class ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def slow_status(client, file_name):
    time.sleep(16)
    return SimpleNamespace(
        file_name=file_name, file_url="https://example.org/probe.zip",
        status="finished", total_rows=7,
    )


async def main() -> None:
    live = sys.argv[1:] == ["live"]
    arguments = {"agency_name": "NSF", "start_date": "2024-01-01", "end_date": "2024-01-31"} if live else {
        "agency_name": "NSF", "start_fiscal_year": 2024, "end_fiscal_year": 2024,
    }
    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "download_records",
            "args": arguments,
            "id": "probe-1",
        }]),
        AIMessage(content="The CSV is ready."),
    ])
    graph = create_react_agent(
        model=model,
        tools=[lc_tool(download_records.func, parse_docstring=False)],
        checkpointer=InMemorySaver(),
    )
    client = SimpleNamespace(
        find_agency_by_name=lambda name: SimpleNamespace(agency_name="National Science Foundation"),
        download_search=lambda filters, columns, levels: SimpleNamespace(
            file_name="probe.zip", status_url="https://example.org/status"
        ),
    )
    started = time.monotonic()
    with (
        patch("backend.app.agent.streaming._get_conversation_graph", return_value=graph),
        patch("backend.app.agent.streaming._is_in_scope", return_value=True),
        patch("backend.app.agent.streaming._looks_like_download_request", return_value=False),
        patch("backend.app.agent.streaming._is_download_followup", return_value=False),
        nullcontext() if live else patch("backend.app.agent.download_tool_spike._get_usaspending_client", return_value=client),
        nullcontext() if live else patch("backend.app.agent.download_tool_spike._build_filters", return_value=SimpleNamespace(time_period=None)),
        nullcontext() if live else patch("backend.app.agent.download_tool_spike._poll_until_finished", side_effect=slow_status),
    ):
        async for frame in sse_event_generator("probe", "probe"):
            print(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 2), "frame": frame.decode().strip()}), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

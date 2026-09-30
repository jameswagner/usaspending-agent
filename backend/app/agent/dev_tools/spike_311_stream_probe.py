"""Exercise streaming SSE frames while a graph tool step blocks."""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from backend.app.agent.streaming import sse_event_generator


class SlowGraph:
    def get_state(self, config):
        return type("State", (), {"values": {}})()

    def stream(self, inputs, config, stream_mode):
        yield {"agent": {"messages": [AIMessage(content="", tool_calls=[{
            "name": "download_records", "args": {"agency_name": "NSF"}, "id": "probe-1"
        }])]}}
        time.sleep(16)
        yield {"tools": {"messages": [ToolMessage(
            content="CSV download ready: probe.zip", tool_call_id="probe-1", name="download_records"
        )]}}
        yield {"agent": {"messages": [AIMessage(content="The CSV is ready.")]}}


async def main() -> None:
    started = time.monotonic()
    with (
        patch("backend.app.agent.streaming._get_conversation_graph", return_value=SlowGraph()),
        patch("backend.app.agent.streaming._is_in_scope", return_value=True),
        patch("backend.app.agent.streaming._looks_like_download_request", return_value=False),
        patch("backend.app.agent.streaming._is_download_followup", return_value=False),
    ):
        async for frame in sse_event_generator("probe", "probe"):
            print(json.dumps({"elapsed_seconds": round(time.monotonic() - started, 2), "frame": frame.decode().strip()}))


if __name__ == "__main__":
    asyncio.run(main())

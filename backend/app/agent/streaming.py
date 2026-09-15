"""Tool-status SSE streaming for POST /ask/stream (see main.py). Mirrors
orchestrator.py's _ask_langgraph exactly (same scope gate, same
_tool_call_log-driven chart/citation building via _build_result) but drives
the graph with .stream(stream_mode="updates") instead of .invoke(), so a
"Calling search_awards..." / "search_awards: 12 awards found" status is
available to the caller as each tool call happens, not just once at the end.

SqliteSaver (singletons.py's checkpointer) is sync-only, so graph.stream()
must run synchronously - but the FastAPI endpoint needs to be async to
return a StreamingResponse. _run_graph_stream runs on a worker thread
(via the event loop's default executor) and pushes (event_name, payload)
tuples onto a thread-safe queue.Queue; sse_event_generator, running on the
event loop, drains that queue and yields SSE frames as they arrive - a
plain queue.Queue.get() call run off-thread per item, not polling.

If the client disconnects mid-stream, there's no way to abort
_run_graph_stream while it's blocked on a live Anthropic/USASpending call -
it simply finishes and its remaining queue.put() calls become no-ops into
an abandoned queue. This isn't a regression: today's blocking
graph.invoke() (_ask_langgraph) has the same property - an aborted client
doesn't stop the in-flight call there either.
"""
from __future__ import annotations

import asyncio
import json
import logging
import queue
from collections.abc import AsyncIterator

from .orchestrator import NOT_FOUND_MESSAGE, AgentResult, _build_result
from .scope import _is_in_scope
from .singletons import _get_conversation_graph
from .tools import _tool_call_log

logger = logging.getLogger(__name__)

# Well under typical proxy/load-balancer idle timeouts (30-60s) - a single
# slow tool call (e.g. usaspending_client.py's Retry with backoff_factor=1.0
# working through 3 retries) can itself take tens of seconds with no event
# to send, so the connection needs its own liveness signal in that gap.
_KEEPALIVE_INTERVAL_SECONDS = 15.0

# Every data tool's `except USASpendingAPIError as e: return f"This query
# failed: {e}."` pattern (spending.py, awards.py, recipients.py, and now
# location.py) is the only signal available here that a ToolMessage is an
# error vs. a normal result - _record_tool_call (and so _tool_call_log) is
# never populated on this path, only on success, so it can't be used to
# distinguish the two.
_TOOL_ERROR_PREFIX = "This query failed:"

_SUMMARY_MAX_CHARS = 200


def _summarize_tool_result(content: str) -> str:
    """A short, human-readable one-liner for the tool_result SSE event -
    the full structured result isn't sent over the wire just to be
    discarded client-side. Strips _wrap_untrusted's <untrusted_data>
    wrapper (tools/_shared.py) so the summary shown to the end user doesn't
    include it.
    """
    text = content
    prefix, suffix = "<untrusted_data>\n", "\n</untrusted_data>"
    if text.startswith(prefix) and text.endswith(suffix):
        text = text[len(prefix) : -len(suffix)]
    text = " ".join(text.split())
    if len(text) > _SUMMARY_MAX_CHARS:
        text = text[:_SUMMARY_MAX_CHARS].rstrip() + "…"
    return text


def _build_done_payload(result: AgentResult) -> dict:
    """Same shape/field logic as main.py's /ask handler builds into
    AskResponse - kept as a plain dict here (not that pydantic model)
    so this module doesn't import from main.py."""
    source_type = "not_found" if result.answer_text == NOT_FOUND_MESSAGE else "agent"
    return {
        "answer_text": result.answer_text,
        "source_type": source_type,
        "conversation_id": result.conversation_id,
        "charts": [c.model_dump() for c in result.charts],
        "citations": [c.model_dump() for c in result.citations],
        "tool_citations": [c.model_dump() for c in result.tool_citations],
    }


def _run_graph_stream(question: str, conversation_id: str, event_queue: queue.Queue) -> None:
    """Runs on a worker thread - see module docstring. Always ends with a
    single None sentinel so the consumer knows to stop, even on failure.
    """
    try:
        graph = _get_conversation_graph()
        config = {"configurable": {"thread_id": conversation_id}}
        recent_messages = graph.get_state(config).values.get("messages", [])

        if not _is_in_scope(question, recent_messages):
            logger.info("Scope gate rejected question: %r", question)
            result = AgentResult(answer_text=NOT_FOUND_MESSAGE, conversation_id=conversation_id)
            event_queue.put(("done", _build_done_payload(result)))
            return

        _tool_call_log.set([])
        last_ai_message = None

        for update in graph.stream(
            {"messages": [{"role": "user", "content": question}]}, config=config, stream_mode="updates"
        ):
            for node, data in update.items():
                messages = (data or {}).get("messages")
                if not messages:
                    continue
                if node == "agent":
                    last_ai_message = messages[-1]
                    for tool_call in getattr(last_ai_message, "tool_calls", None) or []:
                        event_queue.put(
                            ("tool_call_start", {"tool_name": tool_call["name"], "args": tool_call["args"]})
                        )
                elif node == "tools":
                    for message in messages:
                        content = message.content if isinstance(message.content, str) else str(message.content)
                        if content.startswith(_TOOL_ERROR_PREFIX):
                            event_queue.put(("tool_error", {"tool_name": message.name, "message": content}))
                        else:
                            event_queue.put(
                                ("tool_result", {"tool_name": message.name, "summary": _summarize_tool_result(content)})
                            )

        answer_text = last_ai_message.content if last_ai_message is not None else ""
        result = _build_result(answer_text, conversation_id)
        event_queue.put(("done", _build_done_payload(result)))
    except Exception as e:
        logger.exception("Streaming agent turn failed for question: %r", question)
        event_queue.put(("error", {"message": str(e)}))
    finally:
        event_queue.put(None)


def _format_sse_frame(event_name: str, payload: dict) -> bytes:
    """One SSE frame: `event: <name>\\ndata: <json>\\n\\n`. json.dumps
    escapes any embedded newline inside a string field, so a multi-line
    tool summary/error message still round-trips as a single `data:` line
    rather than breaking SSE framing."""
    return f"event: {event_name}\ndata: {json.dumps(payload)}\n\n".encode()


_KEEPALIVE_FRAME = b": keep-alive\n\n"


async def sse_event_generator(question: str, conversation_id: str) -> AsyncIterator[bytes]:
    """The async side of the bridge - drains event_queue, run off the event
    loop's thread per item (a plain blocking Queue.get(), not polling), and
    yields each item as a formatted SSE frame."""
    event_queue: queue.Queue = queue.Queue()
    loop = asyncio.get_running_loop()
    # Fire-and-forget: not awaited here, so the generator can start
    # yielding queue items as soon as the first one arrives rather than
    # waiting for the whole turn to finish. See module docstring for what
    # happens to this future if the client disconnects before it resolves.
    loop.run_in_executor(None, _run_graph_stream, question, conversation_id, event_queue)

    while True:
        try:
            item = await loop.run_in_executor(None, event_queue.get, True, _KEEPALIVE_INTERVAL_SECONDS)
        except queue.Empty:
            yield _KEEPALIVE_FRAME
            continue
        if item is None:
            break
        event_name, payload = item
        yield _format_sse_frame(event_name, payload)

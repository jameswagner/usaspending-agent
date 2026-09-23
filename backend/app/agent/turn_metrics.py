"""Per-turn cost/latency instrumentation for the live agent path (#245).

One structured `logger.info` line per turn via log_turn(), built from a
ModelToolTimingCallback (attached to graph.invoke()/graph.stream() via
LangChain's callback system, which fires identically regardless of which
of those two entry points is used) plus small `timed()`-wrapped sections
for the scope gate and download pilot. Deliberately not a metrics
backend - Railway logs are the consumer.
"""
from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("agent.turn_metrics")


@contextmanager
def timed(sink: dict[str, float] | None, key: str):
    """Records elapsed wall time (ms) into sink[key], if a sink was given."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if sink is not None:
            sink[key] = round((time.perf_counter() - start) * 1000, 1)


class ModelToolTimingCallback(BaseCallbackHandler):
    """Per-model-call and per-tool-call latency/token counts over one
    graph run. CallbackManager isolates exceptions raised by a handler
    (logs, doesn't propagate) by default, so a bug here can't fail a turn.
    """

    def __init__(self) -> None:
        self.model_calls: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._model_starts: dict[Any, float] = {}
        self._tool_starts: dict[Any, tuple[float, str]] = {}

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs) -> None:
        self._model_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kwargs) -> None:
        start = self._model_starts.pop(run_id, None)
        entry: dict[str, Any] = {
            "duration_ms": round((time.perf_counter() - start) * 1000, 1) if start is not None else None
        }
        generations = response.generations or []
        message = getattr(generations[0][0], "message", None) if generations and generations[0] else None
        usage = getattr(message, "usage_metadata", None) if message else None
        if usage:
            details = usage.get("input_token_details") or {}
            entry.update(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
                cache_read_tokens=details.get("cache_read", 0),
                cache_creation_tokens=details.get("cache_creation", 0),
            )
        self.model_calls.append(entry)

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs) -> None:
        self._tool_starts[run_id] = (time.perf_counter(), (serialized or {}).get("name", "unknown_tool"))

    def _finish_tool(self, run_id) -> None:
        start = self._tool_starts.pop(run_id, None)
        if start is None:
            return
        started_at, name = start
        self.tool_calls.append({"tool_name": name, "duration_ms": round((time.perf_counter() - started_at) * 1000, 1)})

    def on_tool_end(self, output, *, run_id, **kwargs) -> None:
        self._finish_tool(run_id)

    def on_tool_error(self, error, *, run_id, **kwargs) -> None:
        self._finish_tool(run_id)

    def token_totals(self) -> dict[str, int]:
        return {
            "total_input_tokens": sum(c.get("input_tokens") or 0 for c in self.model_calls),
            "total_output_tokens": sum(c.get("output_tokens") or 0 for c in self.model_calls),
            "total_cache_read_tokens": sum(c.get("cache_read_tokens") or 0 for c in self.model_calls),
            "total_cache_creation_tokens": sum(c.get("cache_creation_tokens") or 0 for c in self.model_calls),
        }


def log_turn(
    *,
    outcome: str,
    total_ms: float,
    human_turn_count: int,
    scope_timing: dict[str, float] | None = None,
    download_timing: dict[str, float] | None = None,
    metrics: ModelToolTimingCallback | None = None,
    tool_call_count: int = 0,
    tool_call_budget_hit: bool = False,
) -> None:
    """Non-gating: a failure to read usage metadata must never fail the turn."""
    try:
        payload: dict[str, Any] = {"outcome": outcome, "total_ms": total_ms, "human_turn_count": human_turn_count}
        if scope_timing:
            payload["scope_gate"] = scope_timing
        if download_timing:
            payload["download_pilot"] = download_timing
        if metrics is not None:
            payload["model_calls"] = metrics.model_calls
            payload["model_call_count"] = len(metrics.model_calls)
            payload["tool_calls"] = metrics.tool_calls
            payload["tool_call_count"] = tool_call_count
            payload["tool_call_budget_hit"] = tool_call_budget_hit
            payload.update(metrics.token_totals())
        logger.info("agent_turn %s", json.dumps(payload, default=str))
    except Exception:
        logger.exception("Failed to log turn metrics")

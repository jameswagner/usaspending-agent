"""ContextVar-based capture of real HTTP requests made by the client."""
from __future__ import annotations

import contextvars

# Real (method, url, body) for every live HTTP call this client makes -
# used by agent/tools/_shared.py to build reproducible citations. Capturing
# what was actually sent here is correct by construction; reconstructing it
# from a hand-picked field list at the citation layer is what caused a real
# bug once already (a naics_code-only query's citation silently dropped
# naics_code because that list was never updated when it was added).
_request_log: contextvars.ContextVar[list[tuple[str, str, dict | None]] | None] = contextvars.ContextVar(
    "usaspending_request_log", default=None
)


def _record_request(method: str, url: str, body: dict | None) -> None:
    log = _request_log.get()
    if log is None:
        log = []
        _request_log.set(log)
    log.append((method, url, body))


def drain_request_capture() -> list[tuple[str, str, dict | None]]:
    """Returns and clears every request recorded since the last drain."""
    log = _request_log.get() or []
    _request_log.set([])
    return log

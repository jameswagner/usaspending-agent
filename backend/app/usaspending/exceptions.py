"""Error types for the USASpending API client."""
from __future__ import annotations

import requests


class USASpendingAPIError(Exception):
    """Raised on a non-2xx response, with the API's own error detail (if any)
    as the message instead of a raw requests traceback — callers (e.g. an
    LLM tool wrapper) can surface str(e) directly without leaking internals.
    """


# Also raised (not just non-2xx responses) when the session's mounted Retry
# (total=3, backoff_factor=1.0, status_forcelist=[429,500,502,503,504])
# exhausts, or a connection never completes at all - without this, that case
# surfaced as a raw requests exception no caller here catches, so a slow/
# unreachable USASpending.gov crashed the tool call instead of producing an
# honest, model- and user-facing message.
_TIMEOUT_MESSAGE = (
    "USASpending.gov is responding slowly or is temporarily unreachable "
    "(request timed out after retries). Please try again in a moment."
)


def _raise_with_detail(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = None
        try:
            detail = resp.json().get("detail")
        except ValueError:
            pass
        message = detail or str(e)
        raise USASpendingAPIError(f"{resp.status_code}: {message}") from e

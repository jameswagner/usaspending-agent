"""The raw-API agent's only two data tools: read a contract, call an endpoint."""
from __future__ import annotations

import contextvars
import json
from typing import Literal

import requests
from anthropic import beta_tool
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from backend.app.agent.singletons import _get_usaspending_client
from backend.app.agent.tools._shared import _wrap_untrusted

from .contracts import match_endpoint, read_doc

BASE_URL = "https://api.usaspending.gov"
MAX_RESPONSE_CHARS = 15_000
MAX_CONTRACT_CHARS = 40_000

_api_call_log: contextvars.ContextVar[list[dict] | None] = contextvars.ContextVar("raw_api_call_log", default=None)

_session = requests.Session()
_session.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=2, backoff_factor=1, status_forcelist=(502, 503, 504), allowed_methods=None)),
)


def drain_api_calls() -> list[dict]:
    calls = _api_call_log.get() or []
    _api_call_log.set([])
    return calls


def _record(entry: dict) -> None:
    log = _api_call_log.get()
    if log is None:
        log = []
        _api_call_log.set(log)
    log.append(entry)


_AGENCY_LIST_PATH = "/api/v2/references/toptier_agencies/"
_AGENCY_LIST_FIELDS = ("agency_name", "abbreviation", "toptier_code", "agency_id")


def _trim_agency_list(payload: dict) -> dict:
    return {"results": [{k: a.get(k) for k in _AGENCY_LIST_FIELDS} for a in payload.get("results", [])]}


def _fill_agency_code(path: str, agency: str) -> tuple[str | None, str | None]:
    """Returns (path with {toptier_code} filled in, error message)."""
    match = _get_usaspending_client().find_agency_by_name(agency)
    if match is None:
        return None, f"No federal agency matched {agency!r}. Try the agency's full name or its abbreviation."
    return path.replace("{toptier_code}", match.toptier_code), None


def _clip(text: str, limit: int, hint: str) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: showing {limit:,} of {len(text):,} chars. {hint}]"


@beta_tool
def get_api_contract(doc_id: str) -> str:
    """Read the full upstream contract for one endpoint: its request
    parameters, enum members, response fields, and examples. Read it before
    the first call to any endpoint you haven't read the contract for in this
    conversation.

    Args:
        doc_id: A doc_id from the endpoint catalog (e.g.
            "contracts/v2/search/spending_by_category.md"), or
            "search_filters.md" for the shared `filters` object every
            /api/v2/search/* endpoint takes.
    """
    text = read_doc(doc_id)
    if text is None:
        return f"No contract doc named {doc_id!r}. Use a doc_id exactly as listed in the endpoint catalog."
    return _clip(text, MAX_CONTRACT_CHARS, "The rest of this contract is omitted.")


@beta_tool
def call_usaspending_api(
    method: Literal["GET", "POST"],
    path: str,
    body: dict | None = None,
    query_params: dict | None = None,
    agency: str | None = None,
) -> str:
    """Call one live, read-only USAspending API endpoint and return its raw
    JSON response (or its error message). Only endpoints in the catalog are
    callable; download/bulk-download endpoints are not.

    Args:
        method: "GET" or "POST", as the endpoint's contract specifies.
        path: The endpoint path with any path parameters filled in, e.g.
            "/api/v2/agency/080/budgetary_resources/". No host, no query string.
            For an agency endpoint, leave `{toptier_code}` in the path and set `agency`.
        body: JSON request body for POST endpoints.
        query_params: Query-string parameters for GET endpoints.
        agency: An agency name or abbreviation (e.g. "NSF", "National Science Foundation", "NIH").
            Fills `{toptier_code}` in `path` with the agency's code; never guess a code yourself.
    """
    if agency is not None:
        path, error = _fill_agency_code(path, agency)
        if error is not None:
            return error
    path = path.split("?", 1)[0]
    if not path.endswith("/"):
        path += "/"
    doc_id = match_endpoint(method, path)
    if doc_id is None:
        return (
            f"{method} {path} is not a callable endpoint. Use a method and path from the endpoint catalog, "
            "with path parameters filled in."
        )

    url = BASE_URL + path
    try:
        response = _session.request(method, url, json=body if method == "POST" else None, params=query_params, timeout=60)
    except requests.RequestException as exc:
        _record({"method": method, "url": url, "body": body, "params": query_params, "status": None, "doc_id": doc_id})
        return f"Request failed before a response arrived: {type(exc).__name__}: {exc}"

    _record({"method": method, "url": url, "body": body, "params": query_params, "status": response.status_code, "doc_id": doc_id})
    if response.status_code >= 400:
        return f"HTTP {response.status_code} from {method} {path}:\n" + _clip(response.text, 3_000, "")

    try:
        data = response.json()
        if path == _AGENCY_LIST_PATH:
            data = _trim_agency_list(data)
        payload = json.dumps(data, separators=(",", ":"))
    except ValueError:
        payload = response.text
    hint = "Narrow the request (lower `limit`, tighter filters, or fewer `fields`) to see everything."
    return _wrap_untrusted(_clip(payload, MAX_RESPONSE_CHARS, hint))

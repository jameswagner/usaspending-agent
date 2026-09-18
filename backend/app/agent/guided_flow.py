"""The first guided flow: a state + fiscal-year-range spending lookup,
optionally narrowed to one congressional district, shown as separate
prime-award and subaward totals - never summed, since they can overlap.
Modeled directly on USASpending's own "Quickstart" and "Tutorial" videos on
finding spending to your state
(https://www.youtube.com/watch?v=b9ABwzIyCNI,
https://www.youtube.com/watch?v=ZuvZQ33ZvAE), which walk through exactly
this download-and-sum workflow by hand; this flow answers the same question
in one step instead.

Location type is always place of performance, never recipient location
(the awardee's business address), even though the videos' own default
calculation uses recipient location. This isn't a simplification - it's
confirmed to be the only location type this endpoint actually supports.
GitHub issue fedspendingtransparency/usaspending-api#2372 ("Spending by
Category Doesn't Filter") is a maintainer confirming exactly this, for
category="county": "Spending By Category is actually providing PPoP
[place of performance] location data" regardless of which location filter
you pass. Live-confirmed here that the same holds for category="district"
and category="recipient": filtering `recipient_locations` to a single
state returned results labeled with dozens of other states (e.g. asking
for Illinois returned FL-19, NE-01, NC-03, TX-30... as top rows), and
`recipient_locations`+district_current asking for IL-01 surfaced IL-11,
MD-02, DC-98 instead. Not a bug in this codebase or a live-only quirk -
this is the documented, intended behavior of this endpoint family.

"Note that you can use any geographic unit... such as county or state" -
district is one option, not the only one. Leaving district blank here
returns the full per-district breakdown for the state (confirmed live:
that's what the API itself returns when district_current is omitted from
the location filter) rather than requiring one district up front. County
isn't wired in yet - it needs the existing resolve_county_fips tool as an
extra resolution step, a separate piece of work from this file.

Calls the tools' `_raw` functions directly (never the `@beta_tool`-wrapped
versions, never added to LANGGRAPH_TOOLS) - a flow step is a deterministic
lookup with known inputs, not something that needs an LLM to choose a tool
or fill its parameters. `_record_tool_call`/`_tool_call_log` (tools/_shared.py)
are `ask()`-specific plumbing (a no-op outside its ContextVar) so citations
here are built directly via `build_tool_citation` instead, from the same
context-dict shape `_record_tool_call` would have produced.
"""
from __future__ import annotations

import json
import logging

from backend.app.usaspending_client import (
    SpendingByCategoryResponse,
    USASpendingAPIError,
    drain_request_capture,
)

from .response_shaping import ToolCitation, build_tool_citation
from .singletons import MODEL, _get_client
from .tools.spending import get_spending_by_category_raw

logger = logging.getLogger(__name__)

# In-process only - lost on restart, and wrong under multiple backend
# workers (same class of caveat issues #44/#45 document for other lazy
# singletons). Upgrade path if this ever needs to survive restarts/scale
# to multiple workers: a sibling table in conversations.db, which
# singletons.py already owns a raw sqlite3 connection to.
_FLOW_STATE: dict[str, dict] = {}

# district is deliberately not required - blank means "show every district
# in the state" (see module docstring).
_REQUIRED_FIELDS = ("state", "start_fiscal_year", "end_fiscal_year")
_ALL_FIELDS = (*_REQUIRED_FIELDS, "district")

_TOP_N_RECIPIENTS = 5

# Small, fixed vocabulary this flow's own fields/output can raise questions
# about - answered inline without a live tool/RAG call, since v1's topic set
# is small and stable. A topic outside this set falls through as "escape"
# from the classifier below, reaching search_guide via the full agent instead.
_ASIDE_EXPLANATIONS = {
    "subaward": (
        "A subaward is money a prime awardee passes on to another organization "
        "to do part of the work. Prime and subaward totals are reported "
        "separately because they can overlap - adding them together would "
        "double-count some of the same federal dollars."
    ),
    "obligation": (
        "An obligation is a legal commitment to spend money (like signing a "
        "contract), not necessarily money that's already been paid out."
    ),
    "fiscal_year": (
        "The federal fiscal year runs October 1 through September 30, not "
        "January through December - FY2023 means Oct. 1, 2022 - Sep. 30, 2023."
    ),
    "place_of_performance": (
        "This looks up spending by where the work was actually performed, "
        "not the awardee's business address - this flow always uses place "
        "of performance (see this module's own docstring for why recipient "
        "location isn't a real option here)."
    ),
}

FLOW_CLASSIFIER_PROMPT = (
    "You manage one step of a guided data-lookup flow inside a USASpending.gov "
    "chat assistant. The flow collects: state (2-letter code), an optional "
    "congressional district (a number like \"01\", or \"AL\"/blank for an "
    "at-large or whole-state view), start_fiscal_year, and end_fiscal_year. "
    "It reports federal spending totals (by place of performance) for that "
    "scope. Given the current fields and the user's new message, respond "
    "with ONLY a JSON object, no other text, of this exact shape:\n"
    '{"intent": "slot_update" | "aside" | "escape", '
    '"fields": {"state": string or null, "district": string or null, '
    '"start_fiscal_year": integer or null, "end_fiscal_year": integer or null}, '
    '"aside_topic": "subaward" | "obligation" | "fiscal_year" | '
    '"place_of_performance" or null}\n\n'
    '"slot_update": the message supplies or changes one or more fields. '
    "Extract every field you can find in the message, even ones already "
    "filled (e.g. \"actually make that IL-01\" changes both state and "
    "district in one message). A message that only names a state with no "
    "district mention should leave district null, not guess one.\n"
    '"aside": the message asks what one of the flow\'s own concepts means '
    "rather than supplying a value - set aside_topic to the closest match.\n"
    '"escape": anything else - a different spending question, on-topic or '
    "not. If you are unsure between escape and the other two, choose escape."
)


def classify_flow_input(message: str, fields: dict) -> dict:
    """Cheap classifier deciding what one chat message means to an active
    flow. Same inline-raw-client-call pattern as scope.py's _is_in_scope -
    no shared "small structured classification call" helper exists yet in
    this codebase.

    Any parse failure or unrecognized intent defaults to "escape" - the
    fail-safe direction, since escape routes to the real agent loop rather
    than risking a wrong guess about the user's intent.
    """
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=200,
        system=FLOW_CLASSIFIER_PROMPT,
        messages=[{"role": "user", "content": f"Current fields: {json.dumps(fields)}\nMessage: {message}"}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        # Despite the prompt saying "ONLY a JSON object, no other text",
        # confirmed live the model sometimes wraps it in a ```json fence
        # anyway - slice out the outermost {...} rather than trust the
        # response to be bare JSON.
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no JSON object found in response")
        parsed = json.loads(text[start : end + 1])
        if parsed.get("intent") not in ("slot_update", "aside", "escape"):
            raise ValueError(f"unrecognized intent {parsed.get('intent')!r}")
        return parsed
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        logger.warning("classify_flow_input failed to parse %r: %s", text, e)
        return {"intent": "escape", "fields": {}, "aside_topic": None}


def _normalize_district(district: str | None) -> str | None:
    """"AL"/"at-large" (any casing/spacing) means the same thing as the
    API's own "00" code for a state's single at-large district - confirmed
    live 2026-09-17 that the literal string "AL" is NOT itself a valid
    district code (it silently returns empty results, the same trap as a
    wrong number), so this must be normalized before ever reaching the API,
    not passed through."""
    if not district:
        return district
    normalized = district.strip().upper().replace("-", "").replace(" ", "")
    if normalized in ("AL", "ATLARGE"):
        return "00"
    return district


def _location_kwargs(state: str, district: str | None) -> dict:
    """Always place of performance - see this module's docstring for why
    recipient location isn't a real option for this endpoint."""
    kwargs: dict = {"performed_in_state": state}
    if district:
        kwargs["performed_in_district"] = district
    return kwargs


def _cite(fields: dict, location_kwargs: dict, response, category: str = "district") -> ToolCitation | None:
    context = {
        "category": category,
        "start_fiscal_year": fields["start_fiscal_year"],
        "end_fiscal_year": fields["end_fiscal_year"],
        **location_kwargs,
    }
    requests_made = drain_request_capture()
    if requests_made:
        context["_requests"] = requests_made
    return build_tool_citation("get_spending_by_category", context, result=response)


def _lookup(
    fields: dict, location_kwargs: dict, spending_level: str, category: str = "district", limit: int = 5
) -> tuple[SpendingByCategoryResponse, ToolCitation | None]:
    response = get_spending_by_category_raw(
        category=category, agency_name=None,
        start_fiscal_year=fields["start_fiscal_year"], end_fiscal_year=fields["end_fiscal_year"],
        spending_level=spending_level, limit=limit,
        **location_kwargs,
    )
    return response, _cite(fields, location_kwargs, response, category)


def _top_recipients(fields: dict, location_kwargs: dict) -> tuple[list[dict], list[dict], list[ToolCitation]]:
    """Top prime recipients and top subrecipients for this same scope - the
    natural "okay but who's actually getting it" follow-up. Confirmed live
    2026-09-17 that category="recipient" with spending_level="subawards"
    returns SUBrecipient names (not the prime), matching category=
    "district"'s own transactions-vs-subawards split."""
    prime, prime_citation = _lookup(fields, location_kwargs, "transactions", category="recipient", limit=_TOP_N_RECIPIENTS)
    sub, sub_citation = _lookup(fields, location_kwargs, "subawards", category="recipient", limit=_TOP_N_RECIPIENTS)
    top_prime = [{"name": r.name or r.code or "(unnamed)", "amount": r.amount} for r in prime.results]
    top_sub = [{"name": r.name or r.code or "(unnamed)", "amount": r.amount} for r in sub.results]
    return top_prime, top_sub, [c for c in (prime_citation, sub_citation) if c is not None]


def _district_breakdown(
    fields: dict, location_kwargs: dict
) -> tuple[dict[str, dict], list[ToolCitation]]:
    """{code: {code, name, prime_total, subaward_total}} for every district
    matching location_kwargs (no district in it - state-only). Shared by
    _breakdown (shows the whole thing)."""
    # A state can have 50+ districts (California) plus rollup rows for
    # awards not attributable to one district (e.g. "IL-MULTIPLE
    # DISTRICTS") - 60 comfortably covers every real case without the
    # default limit=5 silently truncating most of the state away.
    prime, prime_citation = _lookup(fields, location_kwargs, "transactions", limit=60)
    subaward, subaward_citation = _lookup(fields, location_kwargs, "subawards", limit=60)

    # Confirmed live 2026-09-18: a subaward-only row can carry both code and
    # name as null (a subrecipient whose district couldn't be attributed) -
    # "r.name or r.code" alone still leaves None in that case, which is
    # unsafe as a dict/React key. Substitute a real, stable label instead.
    def _key_and_label(r) -> tuple[str, str]:
        code = r.code or "unattributed"
        name = r.name or r.code or "Unattributed"
        return code, name

    by_code: dict[str, dict] = {}
    for r in prime.results:
        code, name = _key_and_label(r)
        by_code.setdefault(code, {"code": code, "name": name, "prime_total": 0.0, "subaward_total": 0.0})
        by_code[code]["prime_total"] = r.amount
    for r in subaward.results:
        code, name = _key_and_label(r)
        by_code.setdefault(code, {"code": code, "name": name, "prime_total": 0.0, "subaward_total": 0.0})
        by_code[code]["subaward_total"] = r.amount
    return by_code, [c for c in (prime_citation, subaward_citation) if c is not None]


_AT_LARGE_NOTE = (
    '{state} has a single at-large congressional district (coded "00", '
    "not a number) - showing that district's totals."
)


def _single_district(fields: dict) -> dict:
    state, district = fields["state"], fields["district"]
    location_kwargs = _location_kwargs(state, district)

    prime, prime_citation = _lookup(fields, location_kwargs, "transactions")
    subaward, subaward_citation = _lookup(fields, location_kwargs, "subawards")
    prime_amount = prime.results[0].amount if prime.results else None
    subaward_amount = subaward.results[0].amount if subaward.results else None
    citations = [prime_citation, subaward_citation]

    # A wrong district number doesn't error - it silently returns an empty
    # result, which would otherwise be reported as a confident but wrong
    # "$0". Confirmed live 2026-09-17: single-district states (e.g.
    # Wyoming) use district "00" (at-large), not "01" - a common,
    # reasonable guess. Retry once with "00" before accepting a $0 answer,
    # and disclose the substitution. Not hardcoding a states list here
    # (which redistricting - e.g. Montana regaining a second district in
    # 2022 - would make stale) - this self-corrects for any state this
    # actually applies to.
    note = None
    if prime_amount is None and subaward_amount is None and district != "00":
        retry_fields = {**fields, "district": "00"}
        retry_kwargs = _location_kwargs(state, "00")
        retry_prime, retry_prime_citation = _lookup(retry_fields, retry_kwargs, "transactions")
        retry_subaward, retry_subaward_citation = _lookup(retry_fields, retry_kwargs, "subawards")
        if retry_prime.results or retry_subaward.results:
            fields, location_kwargs = retry_fields, retry_kwargs
            prime_amount = retry_prime.results[0].amount if retry_prime.results else 0.0
            subaward_amount = retry_subaward.results[0].amount if retry_subaward.results else 0.0
            citations = [retry_prime_citation, retry_subaward_citation]
            note = _AT_LARGE_NOTE.format(state=fields["state"])

    top_prime, top_sub, recipient_citations = _top_recipients(fields, location_kwargs)
    citations.extend(recipient_citations)

    return {
        "status": "result",
        "prime_total": prime_amount if prime_amount is not None else 0.0,
        "subaward_total": subaward_amount if subaward_amount is not None else 0.0,
        "top_recipients": top_prime,
        "top_subrecipients": top_sub,
        "fields": fields,
        "note": note,
        "tool_citations": [c for c in citations if c is not None],
    }


def _breakdown(fields: dict) -> dict:
    """No district given - the full per-district breakdown for the state,
    matching what the live API itself returns when district is omitted
    from the location filter (confirmed live 2026-09-17), rather than
    requiring one district up front. The state-wide total a bare "state"
    unit would give (per the tutorial's "state or county" remark) is just
    the sum of this same list - no separate query needed."""
    location_kwargs = _location_kwargs(fields["state"], None)
    by_code, citations = _district_breakdown(fields, location_kwargs)
    districts = sorted(by_code.values(), key=lambda d: d["prime_total"], reverse=True)

    return {
        "status": "breakdown",
        "fields": fields,
        "districts": districts,
        "state_prime_total": sum(d["prime_total"] for d in districts),
        "state_subaward_total": sum(d["subaward_total"] for d in districts),
        "tool_citations": citations,
    }


def compute_step(fields: dict) -> dict:
    """Given whatever fields are filled, return the fullest answer
    supportable right now - never gate strictly field-by-field (the
    "avoid the linear trap" principle: show a result the moment enough
    is known, don't force a fixed field order)."""
    missing = [f for f in _REQUIRED_FIELDS if not fields.get(f)]
    if missing:
        # No "which field is missing" prompt here - the frontend form
        # always shows every field at once, so restating which one is
        # empty would just duplicate what the visible form controls
        # already say. `prompt` stays reserved for something actionable
        # (see the API-error case below).
        return {"status": "collecting", "fields": fields}

    fields = dict(fields)
    fields["district"] = _normalize_district(fields.get("district"))

    # Primes the request-capture ContextVar with a real list object before
    # the first @traceable-wrapped raw call below. Confirmed live: without
    # this, the first call's _record_request rebinds the ContextVar (from
    # its default None) *inside* @traceable's copied context, which never
    # propagates back out - drain_request_capture() here would silently see
    # nothing and the first citation would have no curl. Once the var is
    # already bound to a real list, later calls only append to that shared
    # object in place, which does propagate. This is the same reason
    # _check_tool_call_budget (tools/_shared.py) unconditionally drains at
    # the top of every @beta_tool wrapper before its own raw call runs -
    # not just to discard stale leftovers as documented there, but
    # incidentally priming this exact mechanism for the normal tool path.
    drain_request_capture()

    try:
        if fields["district"]:
            return _single_district(fields)
        return _breakdown(fields)
    except USASpendingAPIError as e:
        logger.warning("guided_flow lookup failed for %s: %s", fields, e)
        return {
            "status": "collecting",
            "prompt": f"That didn't work: {e}. Try a different state/district.",
            "fields": fields,
        }


def start_flow(conversation_id: str) -> dict:
    _FLOW_STATE[conversation_id] = {"fields": {}, "paused": False}
    return compute_step({})


def step(conversation_id: str, message: str | None) -> dict:
    """The single entry point the /guided-flow endpoint calls.

    message=None resumes an existing flow (or starts a fresh one if none
    exists) without reclassifying anything - used for the "still want your
    district lookup?" resume affordance after an escape."""
    state = _FLOW_STATE.get(conversation_id)
    if state is None:
        return start_flow(conversation_id)

    if message is None:
        state["paused"] = False
        return compute_step(state["fields"])

    fields = state["fields"]
    classification = classify_flow_input(message, fields)
    intent = classification["intent"]

    if intent == "escape":
        state["paused"] = True
        return {"status": "escaped", "forward_question": message}

    if intent == "aside":
        topic = classification.get("aside_topic")
        answer = _ASIDE_EXPLANATIONS.get(
            topic, "I don't have a canned explanation for that yet - try asking me directly."
        )
        return {"status": "aside_answered", "answer": answer, "fields": fields}

    # slot_update
    for key, value in (classification.get("fields") or {}).items():
        if key in _ALL_FIELDS and value not in (None, ""):
            fields[key] = value
    state["paused"] = False
    return compute_step(fields)

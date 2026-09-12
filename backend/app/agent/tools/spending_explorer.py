"""get_spending_explorer_breakdown - whole-of-government obligated
spending via usaspending.gov's own Explore the Data > Spending Explorer
(POST /api/v2/spending/, api_contracts/v2/spending.md). See #30/#99.

A different data lineage from every other spending tool here: File A/B
account data (treasury account/object class/budget classification), not
File C/D award-search data - these numbers are NOT filterable by
award-search semantics and won't line up with get_spending_by_category/
get_spending_over_time/search_awards for the same period.

This is genuinely ONE flexible endpoint, not a fixed drill ladder -
verified live, not assumed, before designing this: `group_by` can be any
of budget_function/budget_subfunction/federal_account/program_activity/
object_class/agency/recipient, and every one of the seven scoping filter
ids below can be combined with any group_by freely (confirmed live e.g.
object_class grouped and ALSO filtered by agency - cross-hierarchy, not
just a parent->child chain). Only exception found: `recipient` times out
live when completely unscoped (too expensive to compute whole-of-
government) - every other group_by works fine unscoped.

`award` is a documented group_by value but deliberately NOT exposed here:
confirmed live to reliably time out (45s+, zero response) both scoped and
unscoped - not a transient blip, checked twice. Re-verify before adding
it rather than assuming this has changed.

The `agency` filter is the one exception to "get an id from a prior
result": it does NOT accept lookup_agency's toptier_code, only this
endpoint's own internal agency id (a group_by="agency" result's `id`
field - not its `code` field, which IS the toptier code and is rejected
outright). Confirmed live: HHS is toptier_code "075" but Spending
Explorer agency id "806" - passing "075" 400s with "Agency ID provided
does not correspond to a toptier agency". Found live 2026-09-11 after
the model, asked for HHS's spending by object class, correctly called
lookup_agency first (this tool's own docstring said to, at the time),
got HHS's toptier_code, passed it as the agency filter, got a clean
rejection from the live API, and gave up on object_class entirely -
falling back to a PSC breakdown from a completely different tool instead
(disclosed to the user, but not actually answering the question asked) -
rather than resolving the right id. See get_spending_explorer_breakdown's
own Args docstring for the agency param, which now tells the model to
resolve this via group_by="agency" instead of lookup_agency.
"""
from __future__ import annotations

import logging
from typing import Literal, get_args

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import SpendingExplorerResponse, USASpendingAPIError

from ..singletons import _get_usaspending_client
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted

logger = logging.getLogger(__name__)

Quarter = Literal["1", "2", "3", "4"]

GroupBy = Literal[
    "budget_function",
    "budget_subfunction",
    "federal_account",
    "program_activity",
    "object_class",
    "agency",
    "recipient",
]

_FILTER_PARAM_NAMES = (
    "agency",
    "budget_function",
    "budget_subfunction",
    "federal_account",
    "object_class",
    "recipient",
    "program_activity",
)

# GroupBy's own Literal args, not a separately-maintained set - the schema
# and this runtime check can't drift apart. Enforced here, not just left
# to the tool schema built from the Literal, because a group_by that
# slips through anyway (a different call path, a looser SDK version, model
# behavior) must fail fast with an actionable message instead of reaching
# client.spending_explorer() and hanging for real - confirmed live that
# group_by="award" specifically times out with zero response after 45s+,
# both scoped and unscoped (see the module docstring).
_SUPPORTED_GROUP_BY = set(get_args(GroupBy))


@traceable(run_type="tool", name="get_spending_explorer_breakdown_raw")
def get_spending_explorer_breakdown_raw(
    group_by: GroupBy,
    fiscal_year: int,
    quarter: Quarter,
    agency: str | None = None,
    budget_function: str | None = None,
    budget_subfunction: str | None = None,
    federal_account: str | None = None,
    object_class: str | None = None,
    recipient: str | None = None,
    program_activity: str | None = None,
) -> SpendingExplorerResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure, before ever reaching the live API if
    group_by isn't one of the supported values (in particular "award",
    a real API value but confirmed to hang) or if group_by="recipient"
    has no scoping filter at all - confirmed live that query times out
    rather than erroring cleanly, so both fail fast with an actionable
    message instead of hanging the whole turn.
    """
    if group_by not in _SUPPORTED_GROUP_BY:
        raise USASpendingAPIError(
            f"group_by={group_by!r} is not supported. Must be one of: "
            f"{', '.join(sorted(_SUPPORTED_GROUP_BY))}. "
            "(group_by='award' is a real API value but confirmed to hang live - "
            "use search_awards or get_award_details for award-level data instead.)"
        )

    filter_kwargs = {
        "agency": agency,
        "budget_function": budget_function,
        "budget_subfunction": budget_subfunction,
        "federal_account": federal_account,
        "object_class": object_class,
        "recipient": recipient,
        "program_activity": program_activity,
    }
    if group_by == "recipient" and not any(filter_kwargs.values()):
        raise USASpendingAPIError(
            "A whole-of-government recipient breakdown with no scoping filter times out live - "
            "narrow it first with at least one of: agency, budget_function, budget_subfunction, "
            "federal_account, object_class, or program_activity."
        )

    client = _get_usaspending_client()
    filters: dict[str, str] = {"fy": str(fiscal_year), "quarter": quarter}
    for name in _FILTER_PARAM_NAMES:
        value = filter_kwargs[name]
        if value is not None:
            filters[name] = value
    return client.spending_explorer(group_by, filters)


def _format_spending_explorer_results(response: SpendingExplorerResponse) -> str:
    if response.total is None:
        lines = [f"No data as of {response.end_date}."]
    else:
        lines = [f"Total obligated: ${response.total:,.2f} (as of {response.end_date})"]
    for r in response.results:
        if r.id is None:
            # SpendingExplorerGeneralUnreportedResponse shape - the gap
            # between the whole-of-government total above and what's
            # actually been reported at this level so far, not a regular
            # named category. Keyed strictly on a real null id (the
            # contract's own shape for this row), not on a missing name -
            # a program_activity row can have a null name for other
            # reasons (confirmed live) without being this gap.
            lines.append(f"  Unreported data: ${r.amount:,.2f} (not yet broken down at this level)")
            continue
        label = r.name or (f"(unlabeled, code {r.code})" if r.code else "(unlabeled)")
        # Recipient results carry the same string in both fields (verified
        # live) - "NAME (code NAME)" would be redundant noise there.
        if r.code and r.name and r.code != r.name:
            label = f"{label} (code {r.code})"
        account_suffix = f", account {r.account_number}" if r.account_number else ""
        lines.append(f"  {label}: ${r.amount:,.2f}{account_suffix}")
    return "\n".join(lines)


@beta_tool
def get_spending_explorer_breakdown(
    group_by: GroupBy,
    fiscal_year: int,
    quarter: Quarter,
    agency: str | None = None,
    budget_function: str | None = None,
    budget_subfunction: str | None = None,
    federal_account: str | None = None,
    object_class: str | None = None,
    recipient: str | None = None,
    program_activity: str | None = None,
) -> str:
    """Whole-of-government obligated spending, grouped by budget_function/budget_subfunction/federal_account/program_activity/object_class/agency/recipient, for one fiscal year through one fiscal quarter — the same view as usaspending.gov's "Explore the Data > Spending Explorer". Use this for "spending by budget function," "spending by object class," "top agencies by whole-of-government obligations," or similar Explorer-style questions — NEVER get_spending_by_category, which has no budget_function/object_class category at all and reports a different, award-level number.

    IMPORTANT — a different data lineage from every other spending tool here: this is whole-of-government account-level data, not award-level data. Its totals will NOT match get_spending_by_category/get_spending_over_time/search_awards for the same period — that's expected, not a bug, since they answer different questions ("what did the government spend on Medicare overall" vs. "what awards did one agency make").

    Any of the seven optional filters below can be combined with any group_by — this is one flexible endpoint, not a fixed drill ladder. For example, group_by="object_class" filtered by agency shows one agency's spending broken down by object class; group_by="federal_account" filtered by budget_function shows that function's accounts. Get a code/id to use as a filter from a prior call's result (its `code` field), never guessed.

    A result with no code represents "Unreported Data" — the live gap between the whole-of-government total and what's actually been reported at this level so far. State it as an unreported gap if present; never fold it silently into a named category or omit it.

    group_by="recipient" REQUIRES at least one of the other filters — an unscoped whole-of-government recipient breakdown times out live. Also, a recipient result's `id` here is NOT the same recipient_id search_recipients/get_recipient_details expect (missing a required level suffix) — never pass it to those tools directly; use search_recipients by name instead for a follow-up on one specific recipient.

    group_by="award" is not supported — confirmed live to time out.

    Args:
        group_by: What to group results by.
        fiscal_year: e.g. 2024 for FY2024. Data is not available before FY2017 Q2.
        quarter: "1", "2", "3", or "4" — includes all quarters up to and including this one. Only a
            CLOSED fiscal quarter has data; the in-progress quarter fails cleanly (confirmed live:
            "Fiscal parameters provided do not belong to a current submission period") rather than
            returning partial data, and the most recently closed quarter isn't available until
            roughly 45 days after it closes either. There's no "current period" to fall back to —
            if a call fails for this reason, try an earlier fiscal_year/quarter rather than
            guessing forward.
        agency: An agency's Spending Explorer id from a prior call's result with group_by="agency"
            (its `id` field) — NOT lookup_agency's toptier_code, and NOT the `code` field on a
            group_by="agency" result either (that IS the toptier code). Confirmed live these are
            different id spaces (e.g. HHS: Spending Explorer id "806", toptier_code "075") and the
            toptier_code is rejected outright ("Agency ID provided does not correspond to a toptier
            agency"). To scope by an agency you don't already have this id for, call this tool once
            with group_by="agency" (no filters) first to look it up.
        budget_function: A budget function's code (e.g. "570"), from a prior call's result.
        budget_subfunction: A budget sub-function's code, from a prior call's result.
        federal_account: A federal account's code, from a prior call's result.
        object_class: An object class's code, from a prior call's result.
        recipient: A recipient's id, from a prior call's result with group_by="recipient" — see the
            id-format warning above before reusing this elsewhere.
        program_activity: A program activity's code, from a prior call's result.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_spending_explorer_breakdown_raw(
            group_by,
            fiscal_year,
            quarter,
            agency=agency,
            budget_function=budget_function,
            budget_subfunction=budget_subfunction,
            federal_account=federal_account,
            object_class=object_class,
            recipient=recipient,
            program_activity=program_activity,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_explorer_breakdown failed: %s", e)
        return f"This query failed: {e}."

    _record_tool_call(
        "get_spending_explorer_breakdown",
        response,
        {
            "group_by": group_by,
            "fiscal_year": fiscal_year,
            "quarter": quarter,
            "agency": agency,
            "budget_function": budget_function,
            "budget_subfunction": budget_subfunction,
            "federal_account": federal_account,
            "object_class": object_class,
            "recipient": recipient,
            "program_activity": program_activity,
        },
    )

    if not response.results:
        return f"No {group_by} data found for FY{fiscal_year} Q{quarter}."

    return _wrap_untrusted(_format_spending_explorer_results(response))

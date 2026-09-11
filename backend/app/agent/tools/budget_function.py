"""get_spending_by_budget_function - whole-of-government obligated spending
grouped by Budget Function, drilling to Budget Sub-Function then Federal
Account, mirroring usaspending.gov's own Explore the Data > Spending
Explorer. See #30: this is File A/B account-level data (treasury account/
object class), a different lineage from every other spending tool here,
which are all File C/D award-level data via search/spending_by_category
et al. - these numbers are NOT filterable by agency/recipient/award-type
the way those are, and won't line up with them for the same period.

Deliberately stops at Federal Account - Program Activity/Object
Class/Recipient/Award (the Explorer's own deeper levels) is real
additional scope, deferred rather than half-built here.
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import SpendingExplorerResponse, USASpendingAPIError

from ..singletons import _get_usaspending_client
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted

logger = logging.getLogger(__name__)

Quarter = Literal["1", "2", "3", "4"]


@traceable(run_type="tool", name="get_spending_by_budget_function_raw")
def get_spending_by_budget_function_raw(
    fiscal_year: int,
    quarter: Quarter,
    budget_function: str | None = None,
    budget_subfunction: str | None = None,
) -> SpendingExplorerResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure.

    Which live "type"/filters to send is inferred from how far the caller
    has already drilled - no budget_function given: the top-level Budget
    Function breakdown (the live Explorer's "general" query); budget_function
    alone: that function's Sub-Functions; both given: that sub-function's
    Federal Accounts. One level per call, matching the live Explorer's own
    click-to-drill-down flow.
    """
    if budget_subfunction is not None and budget_function is None:
        raise USASpendingAPIError("budget_subfunction requires budget_function to also be set.")

    client = _get_usaspending_client()
    filters: dict[str, str] = {"fy": str(fiscal_year), "quarter": quarter}
    if budget_subfunction is not None:
        filters["budget_function"] = budget_function
        filters["budget_subfunction"] = budget_subfunction
        explorer_type = "federal_account"
    elif budget_function is not None:
        filters["budget_function"] = budget_function
        explorer_type = "budget_subfunction"
    else:
        explorer_type = "budget_function"
    return client.spending_explorer(explorer_type, filters)


def _format_budget_function_results(response: SpendingExplorerResponse) -> str:
    if response.total is None:
        lines = [f"No data as of {response.end_date}."]
    else:
        lines = [f"Total obligated: ${response.total:,.2f} (as of {response.end_date})"]
    for r in response.results:
        if r.id is None:
            # SpendingExplorerGeneralUnreportedResponse shape - the gap
            # between the whole-of-government total above and what's
            # actually been reported at this level so far, not a regular
            # named category.
            lines.append(f"  Unreported data: ${r.amount:,.2f} (not yet broken down at this level)")
            continue
        label = f"{r.name} (code {r.code})" if r.code else r.name
        account_suffix = f", account {r.account_number}" if r.account_number else ""
        lines.append(f"  {label}: ${r.amount:,.2f}{account_suffix}")
    return "\n".join(lines)


@beta_tool
def get_spending_by_budget_function(
    fiscal_year: int,
    quarter: Quarter,
    budget_function: str | None = None,
    budget_subfunction: str | None = None,
) -> str:
    """Whole-of-government obligated spending broken down by Budget Function (e.g. Medicare, Social Security, National Defense — the same view as usaspending.gov's "Explore the Data > Spending Explorer"), for one fiscal year through one fiscal quarter. Use this for "spending by budget function" questions — NEVER get_spending_by_category, which has no budget_function category at all and reports a different, award-level number.

    IMPORTANT — a different data lineage from every other spending tool here: this is whole-of-government account-level data, not award-level data. It cannot be filtered by agency, recipient, or award type, and its totals will NOT match get_spending_by_category/get_spending_over_time/search_awards for the same period — that's expected, not a bug, since they answer different questions ("what did the government spend on Medicare overall" vs. "what awards did one agency make").

    Drills down one level per call, matching the live Explorer's own click-through: call with no budget_function for the top-level Budget Function breakdown; pass a result's code as budget_function to see that function's Budget Sub-Functions; pass both budget_function and budget_subfunction to see that sub-function's Federal Accounts. Does not go deeper than Federal Account — no Program Activity/Object Class/Recipient/Award drill-down.

    A result with no code represents "Unreported Data" — the live gap between the whole-of-government total and what's actually been reported at this level so far. State it as an unreported gap if present; never fold it silently into a named category or omit it.

    Args:
        fiscal_year: e.g. 2024 for FY2024. Data is not available before FY2017 Q2.
        quarter: "1", "2", "3", or "4" — includes all quarters up to and including this one. The
            most recently completed quarter's data isn't available until 45 days after it closes.
        budget_function: A budget function's code (e.g. "570"), from a prior call's result, to drill
            into its sub-functions. Omit for the top-level breakdown.
        budget_subfunction: A budget sub-function's code, from a prior call's result, to drill into
            that sub-function's federal accounts. Requires budget_function to also be set.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_spending_by_budget_function_raw(fiscal_year, quarter, budget_function, budget_subfunction)
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_budget_function failed: %s", e)
        return f"This query failed: {e}."

    _record_tool_call(
        "get_spending_by_budget_function",
        response,
        {
            "fiscal_year": fiscal_year,
            "quarter": quarter,
            "budget_function": budget_function,
            "budget_subfunction": budget_subfunction,
        },
    )

    if not response.results:
        return f"No budget function data found for FY{fiscal_year} Q{quarter}."

    return _wrap_untrusted(_format_budget_function_results(response))

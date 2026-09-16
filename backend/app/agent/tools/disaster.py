"""get_disaster_spending_overview - total disaster-relief budget authority and
award obligations/outlays via GET /api/v2/disaster/overview/{?def_codes}, distinct
from spending_by_category's category="defc" grouping of ordinary award search
results. def_codes filtering on the other spending tools lives in tool_filters.py.
"""
from __future__ import annotations

import logging

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    DisasterOverviewResponse,
    USASpendingAPIError,
)

from ..singletons import _get_usaspending_client
from ..tool_filters import _normalize_def_codes
from ._shared import (
    _check_tool_call_budget,
    _record_tool_call,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="get_disaster_spending_overview_raw")
def get_disaster_spending_overview_raw(def_codes: list[str] | None = None) -> DisasterOverviewResponse:
    """Call the API once, return the structured response. Raises USASpendingAPIError
    on failure. No fiscal_year/agency/recipient scoping - the live endpoint takes none."""
    client = _get_usaspending_client()
    resolved_codes = _normalize_def_codes(def_codes) if def_codes else None
    return client.get_disaster_overview(resolved_codes)


def _format_disaster_overview(response: DisasterOverviewResponse, def_codes: list[str] | None) -> str:
    scope = f"DEFC {', '.join(def_codes)}" if def_codes else "all disaster/relief DEFCs combined"
    lines = [
        f"Scope: {scope}",
        f"Total Budget Authority: ${response.total_budget_authority:,.2f}",
    ]
    spending = response.spending
    if spending.total_obligations is not None:
        lines.append(f"Total Obligations: ${spending.total_obligations:,.2f}")
    if spending.total_outlays is not None:
        lines.append(f"Total Outlays: ${spending.total_outlays:,.2f}")
    if spending.award_obligations is not None:
        lines.append(f"Award Obligations: ${spending.award_obligations:,.2f}")
    if spending.award_outlays is not None:
        lines.append(f"Award Outlays: ${spending.award_outlays:,.2f}")
    if response.funding:
        lines.append("Budget authority by DEFC:")
        for row in response.funding:
            lines.append(f"  {row.def_code}: ${row.amount:,.2f}")
    if response.additional is not None:
        lines.append(
            "NOTE: This total also includes an 'additional' figure the API reports separately - "
            f"${response.additional.total_budget_authority:,.2f} in budget authority whose financial "
            "details weren't labeled with the searched DEFC but should still count toward this overview "
            "(already folded into the totals above, not extra on top)."
        )
    return "\n".join(lines)


@beta_tool
def get_disaster_spending_overview(def_codes: list[str] | None = None) -> str:
    """Get the headline disaster/emergency-relief spending numbers: total budget authority, total obligations, total outlays, and the award-specific obligations/outlays subset of those totals — optionally scoped to specific Disaster Emergency Fund Codes (DEFC), e.g. COVID-19 or infrastructure relief. Use this for "how much has the government spent on COVID-19/pandemic relief," "how much infrastructure/IIJA funding has gone out," or "what's the total disaster relief budget" questions — these all-time, government-wide totals aren't answerable from search_awards/get_spending_by_category/get_spending_over_time, which only see individual award records, not the appropriation-level budget authority this tool reports.

    This is a different kind of question from "break existing spending down by DEFC" (get_spending_by_category with category="defc") — this tool reports the disaster-relief-specific budget authority/obligation/outlay figures the real site's own COVID-19/disaster landing page leads with, not a category breakdown of ordinary award search results.

    No fiscal-year or agency/recipient scoping — the live endpoint reports all-time totals only, since disaster relief legislation is inherently government-wide rather than tied to one agency's annual appropriation.

    Args:
        def_codes: Optional. Restrict to these Disaster Emergency Fund Codes (DEFC), e.g.
            ["L"] for a single code. Use the group alias "covid" (or "covid_19") for all
            7 COVID-19-relief codes, or "infrastructure" (or "iija") for both infrastructure
            codes, instead of listing individual letters/numbers — e.g. def_codes=["covid"]
            for "how much has been spent on COVID-19 relief." Omit for the combined total
            across every disaster/relief DEFC ever issued, not just COVID-19/infrastructure.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_disaster_spending_overview_raw(def_codes)
    except USASpendingAPIError as e:
        logger.warning("get_disaster_spending_overview failed for def_codes=%s: %s", def_codes, e)
        return f"This query failed: {e}."

    context: dict = {}
    if def_codes:
        context["def_codes"] = def_codes
    _record_tool_call("get_disaster_spending_overview", response, context)

    return _wrap_untrusted(_format_disaster_overview(response, def_codes))

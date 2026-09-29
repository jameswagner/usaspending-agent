"""Spike #288: prototype consolidated agency-profile tools.

NOT wired into ask() - query_agency_budget/query_agency_award_activity here
are unregistered dispatch prototypes (not imported by langgraph_tools.py or
tools/__init__.py), the same spike-not-implementation status
consolidated.py had for #265 (see d864033/docs/spike-265-consolidate-tools.md,
since removed once #265 went from spike to the real query_spending tool).

Dispatch routes to the *existing* _raw functions for the three lineages
_shared.py already covers, and to the three new client methods added for
this spike (get_agency_sub_component_federal_accounts,
get_agency_obligations_by_award_category, get_agency_awards - #286/#287's
gaps plus the previously-unused whole-agency awards endpoint) for the two
new ones - reusing client.find_agency_by_name rather than re-deriving
resolution, per the issue's own instruction.
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool

from backend.app.usaspending import USASpendingAPIError

from ..singletons import _get_usaspending_client
from ..tool_filters import AwardType
from ._shared import (
    _check_tool_call_budget,
    _format_agency_award_breakdown,
    _format_agency_sub_components,
    _format_api_messages,
    _format_period_breakdown,
    _record_tool_call,
    _truncation_note,
    _wrap_untrusted,
    get_agency_award_breakdown_raw,
    get_agency_budget_by_subcomponent_raw,
    get_agency_budget_raw,
)

logger = logging.getLogger(__name__)

BudgetGroupBy = Literal["sub_component"]
AwardActivityGroupBy = Literal["sub_agency", "award_category"]


def _format_federal_accounts(response) -> str:
    ranked = sorted(response.results, key=lambda r: r.total_obligations, reverse=True)
    lines = [
        f"{r.name}: budgetary resources ${r.total_budgetary_resources:,.2f}, "
        f"obligated ${r.total_obligations:,.2f}, outlayed ${r.total_outlays:,.2f}"
        for r in ranked
    ]
    t = response.totals
    header = (
        f"{response.bureau_slug} totals - budgetary resources ${t.total_budgetary_resources:,.2f}, "
        f"obligated ${t.total_obligations:,.2f}, outlayed ${t.total_outlays:,.2f}"
    )
    return header + "\n" + "\n".join(lines)


def _format_award_category_breakdown(response) -> str:
    ranked = sorted(response.results, key=lambda r: r.aggregated_amount, reverse=True)
    lines = [f"{r.category}: ${r.aggregated_amount:,.2f}" for r in ranked]
    return f"Total: ${response.total_aggregated_amount:,.2f}\n" + "\n".join(lines)


@beta_tool
def query_agency_budget(
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    group_by: BudgetGroupBy | None = None,
    bureau: str | None = None,
    include_period_breakdown: bool = False,
) -> str:
    """Get an agency's actual appropriated budgetary resources, obligations, and outlays. Use this for "what is X's budget" / "how much has X actually paid out" questions, at the whole-agency level (default), broken down by sub-component/bureau (group_by="sub_component"), or for one named bureau's own federal accounts (group_by="sub_component" + bureau).

    Args:
        agency_name: The agency's name, e.g. "National Science Foundation".
        start_fiscal_year: First fiscal year to include. Ignored (must equal end_fiscal_year)
            when group_by is set - the sub-component and bureau breakdowns are single-FY only.
        end_fiscal_year: Last fiscal year to include.
        group_by: None for the whole agency's own yearly totals across the requested range.
            "sub_component" to break the same figures down by sub-component/bureau (e.g. NIH
            or CDC within HHS) for a single fiscal year - requires start_fiscal_year ==
            end_fiscal_year.
        bureau: Only valid with group_by="sub_component". The exact bureau/sub-component name
            as it appears in a prior group_by="sub_component" call's results (e.g. "Food and
            Nutrition Service"), to drill into that one bureau's own federal accounts instead of
            listing every bureau. Omit to list all bureaus.
        include_period_breakdown: Only valid with group_by=None. Set True only when the question
            is specifically about WHEN during the fiscal year money was obligated. Each period's
            obligated amount is CUMULATIVE from the start of the fiscal year, not incremental.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget

    if group_by is None:
        if bureau is not None:
            return "bureau is only valid with group_by=\"sub_component\"."
        try:
            toptier_code, years = get_agency_budget_raw(agency_name, start_fiscal_year, end_fiscal_year)
        except USASpendingAPIError as e:
            logger.warning("query_agency_budget failed for %s: %s", agency_name, e)
            return f"This query failed: {e}."

        _record_tool_call(
            "query_agency_budget",
            years,
            {
                "agency_name": agency_name, "start_fiscal_year": start_fiscal_year,
                "end_fiscal_year": end_fiscal_year, "toptier_code": toptier_code,
            },
        )
        if not years:
            return f"No budget data found for {agency_name} between FY{start_fiscal_year} and FY{end_fiscal_year}."

        def _fmt(amount: float | None) -> str:
            return f"${amount:,.2f}" if amount is not None else "not reported"

        lines = []
        for y in sorted(years, key=lambda y: y.fiscal_year):
            line = (
                f"FY{y.fiscal_year}: budgetary resources {_fmt(y.agency_budgetary_resources)}, "
                f"obligated {_fmt(y.agency_total_obligated)}, outlayed {_fmt(y.agency_total_outlayed)}"
            )
            if include_period_breakdown and y.agency_obligation_by_period:
                line += "\n  " + _format_period_breakdown(y.agency_obligation_by_period)
            lines.append(line)
        return _wrap_untrusted("\n".join(lines))

    # group_by == "sub_component"
    if start_fiscal_year != end_fiscal_year:
        return (
            "group_by=\"sub_component\" only supports a single fiscal year - call again with "
            "start_fiscal_year == end_fiscal_year for each year needed."
        )
    fiscal_year = start_fiscal_year

    if bureau is None:
        try:
            response = get_agency_budget_by_subcomponent_raw(agency_name, fiscal_year)
        except USASpendingAPIError as e:
            logger.warning("query_agency_budget (sub_component) failed for %s: %s", agency_name, e)
            return f"This query failed: {e}."
        _record_tool_call(
            "query_agency_budget",
            response,
            {"agency_name": agency_name, "fiscal_year": fiscal_year, "group_by": "sub_component"},
        )
        if not response.results:
            return f"No sub-component budget data found for {agency_name} in FY{fiscal_year}."
        has_next = response.page_metadata.hasNext if response.page_metadata else False
        note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
        return _wrap_untrusted(_format_agency_sub_components(response) + note)

    # bureau drill-down (#287's gap) - resolve the bureau name to its slug via
    # the parent sub_components list first, since bureau_slug isn't something
    # the model can be expected to already know.
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        return f"No agency found matching '{agency_name}'."
    try:
        parent = client.get_agency_sub_components(agency.toptier_code, fiscal_year=fiscal_year, limit=50)
    except USASpendingAPIError as e:
        logger.warning("query_agency_budget (bureau resolve) failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."
    match = next((r for r in parent.results if bureau.lower() in r.name.lower()), None)
    if match is None:
        names = ", ".join(r.name for r in parent.results)
        return f"No bureau matching '{bureau}' found for {agency_name} in FY{fiscal_year}. Known bureaus: {names}"

    try:
        response = client.get_agency_sub_component_federal_accounts(
            agency.toptier_code, match.id, fiscal_year=fiscal_year, limit=50,
        )
    except USASpendingAPIError as e:
        logger.warning("query_agency_budget (bureau) failed for %s/%s: %s", agency_name, bureau, e)
        return f"This query failed: {e}."
    _record_tool_call(
        "query_agency_budget",
        response,
        {"agency_name": agency_name, "fiscal_year": fiscal_year, "group_by": "sub_component", "bureau": bureau},
    )
    if not response.results:
        return f"No federal account data found for {match.name} in FY{fiscal_year}."
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
    return _wrap_untrusted(_format_federal_accounts(response) + note)


@beta_tool
def query_agency_award_activity(
    agency_name: str,
    fiscal_year: int,
    group_by: AwardActivityGroupBy | None = None,
    award_type: AwardType | None = None,
    include_offices: bool = False,
) -> str:
    """Get one agency's award spending activity (obligations from awards, with transaction/new-award counts) for a single fiscal year - the whole agency's totals (default), broken down by sub-agency (group_by="sub_agency"), or broken down by award category: contracts/idvs/grants/loans/direct_payments/other (group_by="award_category").

    This is a genuinely different lineage from get_spending_by_category/get_spending_over_time/
    search_awards (award-search filters) and from query_agency_budget (appropriated budget
    authority, not award obligations) - use this specifically for agency-profile-style award
    activity questions, always scoped to one agency and one fiscal year.

    Args:
        agency_name: The agency's name, e.g. "Department of Health and Human Services".
        fiscal_year: A single fiscal year - this lineage does not accept a range; call again per
            year for a multi-year view.
        group_by: None for the agency's own whole-agency transaction count and obligations total.
            "sub_agency" to break obligations, transaction counts, and new-award counts down by
            sub-agency. "award_category" to break total obligations down by award category
            (contracts/idvs/grants/loans/direct_payments/other) instead - a different data source
            from get_spending_by_category(category="award_type", ...) despite sharing category
            names; don't treat the two as interchangeable.
        award_type: Only valid with group_by=None or group_by="sub_agency". Restrict to one award
            type or bucket - same vocabulary as search_awards's award_type. Omit to include all.
        include_offices: Only valid with group_by="sub_agency". Set True to also list each
            sub-agency's individual awarding offices.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget

    if group_by == "award_category":
        if award_type is not None:
            return "award_type is not valid with group_by=\"award_category\"."
        if include_offices:
            return "include_offices is only valid with group_by=\"sub_agency\"."
        client = _get_usaspending_client()
        agency = client.find_agency_by_name(agency_name)
        if agency is None:
            return f"No agency found matching '{agency_name}'."
        try:
            response = client.get_agency_obligations_by_award_category(agency.toptier_code, fiscal_year=fiscal_year)
        except USASpendingAPIError as e:
            logger.warning("query_agency_award_activity (award_category) failed for %s: %s", agency_name, e)
            return f"This query failed: {e}."
        _record_tool_call(
            "query_agency_award_activity",
            response,
            {"agency_name": agency_name, "fiscal_year": fiscal_year, "group_by": "award_category"},
        )
        if not response.results:
            return f"No award-category data found for {agency_name} in FY{fiscal_year}."
        return _wrap_untrusted(_format_award_category_breakdown(response) + _format_api_messages(response.messages))

    if group_by == "sub_agency":
        try:
            response = get_agency_award_breakdown_raw(agency_name, fiscal_year, award_type)
        except USASpendingAPIError as e:
            logger.warning("query_agency_award_activity (sub_agency) failed for %s: %s", agency_name, e)
            return f"This query failed: {e}."
        _record_tool_call(
            "query_agency_award_activity",
            response,
            {"agency_name": agency_name, "fiscal_year": fiscal_year, "group_by": "sub_agency", "award_type": award_type},
        )
        if not response.results:
            return f"No award data found for {agency_name} in FY{fiscal_year}."
        has_next = response.page_metadata.hasNext if response.page_metadata else False
        note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
        return _wrap_untrusted(_format_agency_award_breakdown(response, include_offices) + note)

    # group_by is None - whole-agency totals, the previously-unused /awards/ endpoint
    if include_offices:
        return "include_offices is only valid with group_by=\"sub_agency\"."
    client = _get_usaspending_client()
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        return f"No agency found matching '{agency_name}'."
    award_type_codes = None
    if award_type is not None:
        from ..tool_filters import AWARD_TYPE_GROUPS, _normalize_award_type
        award_type_codes = AWARD_TYPE_GROUPS[_normalize_award_type(award_type)]
    try:
        response = client.get_agency_awards(agency.toptier_code, fiscal_year=fiscal_year, award_type_codes=award_type_codes)
    except USASpendingAPIError as e:
        logger.warning("query_agency_award_activity failed for %s: %s", agency_name, e)
        return f"This query failed: {e}."
    _record_tool_call(
        "query_agency_award_activity",
        response,
        {"agency_name": agency_name, "fiscal_year": fiscal_year, "award_type": award_type},
    )
    return _wrap_untrusted(
        f"{agency_name} FY{fiscal_year}: {response.transaction_count:,} transactions, "
        f"${response.obligations:,.2f} in obligations"
        f"{f' (latest action {response.latest_action_date})' if response.latest_action_date else ''}"
        + _format_api_messages(response.messages)
    )

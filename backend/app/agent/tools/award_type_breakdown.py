"""get_award_type_breakdown - the six-way award-type count split
(contracts, contract IDVs, grants, direct payments, loans, other) the
real Advanced Search results page shows first, above any other
breakdown. See #123.

Backed by POST /api/v2/search/spending_by_award_count/
(spending_by_award_count.md), a different endpoint from
spending_by_category (award_type isn't a valid category value there -
confirmed live, see #123's own investigation) and from search_awards
(which returns a ranked/paginated award list, not counts, and has no
total-count field at all).
"""
from __future__ import annotations

import logging

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import SpendingByAwardCountResponse, USASpendingAPIError

from ..recipient_types import RecipientType
from ..singletons import _get_usaspending_client
from ..tool_filters import (
    DateType,
    Scope,
    _build_filters,
    _record_optional_filter_context,
)
from ._shared import (
    _check_tool_call_budget,
    _format_api_messages,
    _record_tool_call,
    _scope_label,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="get_award_type_breakdown_raw")
def get_award_type_breakdown_raw(
    start_fiscal_year: int,
    end_fiscal_year: int,
    agency_name: str | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    performed_in_county: str | None = None,
    recipient_in_county: str | None = None,
    performed_in_city: str | None = None,
    recipient_in_city: str | None = None,
    performed_in_zip: str | None = None,
    recipient_in_zip: str | None = None,
    performed_in_district: str | None = None,
    recipient_in_district: str | None = None,
    keywords: str | None = None,
    date_type: str | None = None,
    place_of_performance_scope: str | None = None,
    recipient_scope: str | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
    award_id: str | None = None,
    recipient_type: str | None = None,
    description: str | None = None,
) -> SpendingByAwardCountResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure.

    No award_type param - this tool answers "how many of each award
    type," so filtering to one type first would defeat its own purpose.

    Unlike every other _build_filters-based tool here, no real scoping
    filter is required - scope_required=False (see tool_filters.py) -
    since spending_by_award_count returns a fixed six-integer shape and
    is confirmed live to answer fast even fully unscoped, unlike the
    timeout risk get_spending_by_category's group_by="recipient"
    documents for an unscoped whole-of-government breakdown.
    """
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county,
        recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city,
        recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip,
        recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district,
        recipient_in_district=recipient_in_district,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
        award_id=award_id,
        recipient_type=recipient_type,
        description=description,
        scope_required=False,
    )
    return client.spending_by_award_count(filters)


def _format_award_type_counts(response: SpendingByAwardCountResponse) -> str:
    counts = response.results
    total = (
        counts.contracts + counts.idvs + counts.grants
        + counts.direct_payments + counts.loans + counts.other
    )
    lines = [
        f"Contracts: {counts.contracts:,}",
        f"Contract IDVs: {counts.idvs:,}",
        f"Grants: {counts.grants:,}",
        f"Direct Payments: {counts.direct_payments:,}",
        f"Loans: {counts.loans:,}",
        f"Other: {counts.other:,}",
        f"Total: {total:,}",
    ]
    return "\n".join(lines) + _format_api_messages(response.messages)


@beta_tool
def get_award_type_breakdown(
    start_fiscal_year: int,
    end_fiscal_year: int,
    agency_name: str | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    performed_in_county: str | None = None,
    recipient_in_county: str | None = None,
    performed_in_city: str | None = None,
    recipient_in_city: str | None = None,
    performed_in_zip: str | None = None,
    recipient_in_zip: str | None = None,
    performed_in_district: str | None = None,
    recipient_in_district: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
    award_id: str | None = None,
    recipient_type: RecipientType | None = None,
    description: str | None = None,
) -> str:
    """Get the count of awards by award type (Contracts, Contract IDVs, Grants, Direct Payments, Loans, Other) for a fiscal year range — the first thing the real Advanced Search results page shows, before any other breakdown. Use this for "how many contracts vs. grants vs. loans," "what's the award-type mix," or any "count/total broken down by award type" question — do NOT reconstruct this by calling search_awards or get_spending_by_category once per award type and adding the results yourself; this tool returns the real, complete six-way split in one call.

    Unlike every other spending tool here, no scoping filter is required — a fully unscoped call (just a fiscal year range) is valid and answers "how many awards of each type, government-wide" directly, matching the real site's own unscoped Advanced Search view. Pass agency_name/recipient_name/etc. only to narrow the split to one agency, recipient, location, industry, etc.

    This tool has no award_type parameter — it answers "how many of each type," so filtering to one type first would defeat the point. For a single type's own detail (e.g. the actual list of grants, or grants' total dollar amount), use search_awards/get_spending_by_category/get_spending_over_time with award_type set instead.

    Args:
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation". Omit for a government-wide split.
        recipient_name: Optional. Restrict to awards whose recipient name contains this text, e.g. "Leidos". An approximate text match — prefer recipient_id when you have one.
        recipient_id: Optional. The exact recipient_id from a prior search_recipients or get_recipient_details call — an exact identifier, not a text match. Do not guess or construct one.
        min_amount: Optional. Restrict to awards of at least this dollar amount.
        max_amount: Optional. Restrict to awards of at most this dollar amount.
        performed_in_state: Optional. Restrict to work performed in this US state, e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to recipients headquartered/located in this US state — different from performed_in_state.
        performed_in_county: Optional. A 3-digit FIPS county code (use resolve_county_fips). Requires performed_in_state also be set.
        recipient_in_county: Optional. Same as performed_in_county, but for the recipient's location. Requires recipient_in_state also be set.
        performed_in_city: Optional. Restrict to work performed in this city, e.g. "Livermore".
        recipient_in_city: Optional. Same as performed_in_city, but for the recipient's location.
        performed_in_zip: Optional. Restrict to work performed in this 5-digit zip code.
        recipient_in_zip: Optional. Restrict to a recipient located in this 5-digit zip code.
        performed_in_district: Optional. A 2-digit congressional district number, paired with performed_in_state.
        recipient_in_district: Optional. Same as performed_in_district, but for the recipient's location, paired with recipient_in_state.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. Which award date the fiscal-year range is matched against — action_date (default), date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" — where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" — where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511".
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants only), format NN.NNN.
        award_id: Optional. Restrict to a single known award (PIID/FAIN/URI) — a fuzzy text match, not an exact-id lookup.
        recipient_type: Optional. Restrict to recipients tagged with this business/recipient type, e.g. "small_business", "nonprofit".
        description: Optional. Restrict to awards whose own description text matches this phrase — distinct from keywords.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    scope = _scope_label(
        agency_name, recipient_name, recipient_id,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
        award_id=award_id, description=description,
    )
    try:
        response = get_award_type_breakdown_raw(
            start_fiscal_year,
            end_fiscal_year,
            agency_name=agency_name,
            recipient_name=recipient_name,
            recipient_id=recipient_id,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            performed_in_county=performed_in_county,
            recipient_in_county=recipient_in_county,
            performed_in_city=performed_in_city,
            recipient_in_city=recipient_in_city,
            performed_in_zip=performed_in_zip,
            recipient_in_zip=recipient_in_zip,
            performed_in_district=performed_in_district,
            recipient_in_district=recipient_in_district,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
            award_id=award_id,
            recipient_type=recipient_type,
            description=description,
        )
    except USASpendingAPIError as e:
        logger.warning("get_award_type_breakdown failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {"start_fiscal_year": start_fiscal_year, "end_fiscal_year": end_fiscal_year},
        agency_name=agency_name,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county,
        recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city,
        recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip,
        recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district,
        recipient_in_district=recipient_in_district,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
        award_id=award_id,
        recipient_type=recipient_type,
        description=description,
    )
    _record_tool_call("get_award_type_breakdown", response, context)
    return _wrap_untrusted(_format_award_type_counts(response))

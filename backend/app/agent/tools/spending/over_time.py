"""get_spending_over_time - spending trends/grand totals grouped by period
(fiscal_year, calendar_year, quarter, month).
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending import SpendingOverTimeResponse, USASpendingAPIError

from ...recipient_types import RecipientType
from ...response_shaping import _format_time_period, year_label
from ...singletons import _get_usaspending_client
from ...tool_filters import (
    AwardType,
    DateType,
    Scope,
    _build_filters,
    _pop_naics_disclosure,
    _record_optional_filter_context,
)
from .._shared import (
    _check_tool_call_budget,
    _format_api_messages,
    _record_tool_call,
    _scope_label,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)

# Per spending_over_time.md's `group` enum - previously unvalidated in
# code at all (BACKLOG.md judged this "safe in practice" since the live
# API's own 400 for a bad value is already clean and informative, unlike
# category's bare 404). Added now anyway for consistency now that every
# other fixed-vocabulary param gets both a schema-level Literal and a
# runtime check - group was the one exception, not because it needed to
# stay one.
VALID_GROUPS = {"fiscal_year", "calendar_year", "quarter", "month"}

Group = Literal["fiscal_year", "calendar_year", "quarter", "month"]


def _normalize_group(group: str) -> str:
    normalized = group.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in VALID_GROUPS:
        raise USASpendingAPIError(
            f"Unknown group '{group}'. Must be one of: {', '.join(sorted(VALID_GROUPS))}"
        )
    return normalized


@traceable(run_type="tool", name="get_spending_over_time_raw")
def get_spending_over_time_raw(
    agency_name: str | None,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    group: Group = "fiscal_year",
    award_type: AwardType | None = None,
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
    def_codes: list[str] | None = None,
) -> SpendingOverTimeResponse:
    """Call the API once, return the structured response. Same filter
    resolution (via _build_filters) as get_spending_by_category_raw -
    including agency_name's optionality and recipient_id's exactness, see
    that function's docstring."""
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        time_period_type,
        start_year,
        end_year,
        award_type=award_type,
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
        def_codes=def_codes,
    )
    return client.spending_over_time(filters, group=_normalize_group(group))


@beta_tool
def get_spending_over_time(
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    group: Group = "fiscal_year",
    agency_name: str | None = None,
    award_type: AwardType | None = None,
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
    def_codes: list[str] | None = None,
) -> str:
    """Get USASpending spending trends over time for a fiscal year range, scoped by a real scoping filter, grouped by period. Use this for "how much/what total funding went to X" questions (group by fiscal_year over the requested range - the response's aggregated_amount for a single-period range is the exact grand total, computed server-side, not a top-N slice) as well as "how has X's spending changed/trended over time" questions.

    At least one real scoping filter must be given - agency_name, recipient_name,
    recipient_id, a location filter (performed_in_*/recipient_in_*), naics_code, psc_code,
    cfda_program, keywords, award_id, description, or recipient_type. A query scoped by
    none of them would mean all federal spending, ever, which this tool refuses rather than
    silently running.

    Args:
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 (Oct 2020-Sep 2021) or CY2021 (Jan-Dec 2021). Data is only
            available from FY2008 (or CY2007) onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        group: One of: fiscal_year, calendar_year, quarter, month. Default fiscal_year. This
            controls how the OUTPUT time series is bucketed - independent of time_period_type,
            which controls how start_year/end_year themselves are interpreted.
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
            Omit for a cross-agency trend for one recipient - but then recipient_name or
            recipient_id must be set instead.
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos". This is an approximate text match - confirmed
            live that it can both miss real subsidiaries whose legal name differs from the
            parent company's, and sweep in unrelated similarly-named entities (e.g. an
            unrelated joint venture). If you already have a specific recipient_id (e.g. from
            search_recipients), pass that instead for an exact, correct trend.
        recipient_id: Optional. The exact recipient_id from a prior search_recipients or
            get_recipient_details call (e.g. "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P") - an
            exact identifier, not a text match. Prefer this over recipient_name whenever you
            have it. Do not guess or construct a recipient_id.
        min_amount: Optional. Restrict to spending of at least this dollar amount.
        max_amount: Optional. Restrict to spending of at most this dollar amount.
        performed_in_state: Optional. Restrict to spending on work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to spending on recipients
            headquartered/located in this US state.
        performed_in_county: Optional. A specific county where work was performed - a 3-digit
            FIPS code (e.g. "025" for Yavapai County, AZ), not a name. Requires
            performed_in_state also be set.
        recipient_in_county: Optional. Same as performed_in_county, but for the recipient's
            location. Requires recipient_in_state also be set.
        performed_in_city: Optional. Restrict to work performed in this city, e.g. "Livermore".
            A plain name, not a code - can match the same city name across every state if
            performed_in_state isn't also set.
        recipient_in_city: Optional. Same as performed_in_city, but for the recipient's location.
        performed_in_zip: Optional. Restrict to work performed in this 5-digit zip code.
        recipient_in_zip: Optional. Restrict to a recipient located in this 5-digit zip code.
        performed_in_district: Optional. A specific congressional district where work was
            performed - a 2-digit number (e.g. "01"), paired with performed_in_state. Reflects
            today's district boundaries, not necessarily the boundaries in effect during a
            historical fiscal year.
        recipient_in_district: Optional. Same as performed_in_district, but for the recipient's
            location, paired with recipient_in_state.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
            Do NOT restate the award_type/category itself here (e.g. "grant", "contracts",
            "cooperative agreement") - award_type already scopes that precisely, and doing so
            on top silently narrows results to only the awards whose description text happens
            to contain that literal word (most awards of that type don't say it) rather than
            broadening or duplicating the filter. Rejected with an error if given alone.
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default: any transaction/modification in the window - note this means
            a multi-year award active in more than one fiscal year contributes to EACH of those
            years' totals), date_signed (the award's original signing date), last_modified_date,
            or new_awards_only (only awards that originated in this window).
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511" - or a
            plain-English industry description (see resolve_naics_code above); if ambiguous, use
            get_spending_by_category with category="naics" to browse instead of guessing.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
        award_id: Optional. Restrict to a single known award (PIID/FAIN/URI), e.g.
            "1605SS17F00018" - a fuzzy text match, not an exact-id lookup.
        recipient_type: Optional. Restrict to recipients tagged with this business/recipient
            type, e.g. "small_business", "woman_owned_business", "nonprofit", "higher_education".
            Sufficient scope on its own, unlike award_type.
        description: Optional. Restrict to awards whose own description text matches this
            phrase, e.g. "vaccine research". Distinct from keywords - keywords also matches
            recipient name, PIID/FAIN/URI, and NAICS/PSC description text, so a keywords hit
            doesn't imply a description hit or vice versa.
        def_codes: Optional. Restrict to spending tagged with these Disaster Emergency Fund
            Codes (DEFC), e.g. ["L"] for a single code. Use the group alias "covid" (or
            "covid_19") or "infrastructure" (or "iija") instead of listing every individual
            code in that group - e.g. def_codes=["covid"] covers all 7 COVID-relief codes.
            Sufficient scope on its own - use this for "how much has been spent on COVID-19
            relief/infrastructure funding over time" trend questions.
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
        award_id=award_id, description=description, def_codes=def_codes,
    )
    try:
        response = get_spending_over_time_raw(
            agency_name,
            time_period_type,
            start_year,
            end_year,
            group,
            award_type=award_type,
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
            def_codes=def_codes,
        )
    except USASpendingAPIError as e:
        _pop_naics_disclosure()
        logger.warning("get_spending_over_time failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    naics_note = _pop_naics_disclosure()
    context = _record_optional_filter_context(
        {
            "start_year": start_year,
            "end_year": end_year,
            "time_period_type": time_period_type,
            "group": group,
        },
        agency_name=agency_name,
        award_type=award_type,
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
        def_codes=def_codes,
    )
    if naics_note:
        context["naics_auto_resolved"] = naics_note
    _record_tool_call("get_spending_over_time", response, context)

    if not response.results:
        no_results_note = f" ({naics_note})" if naics_note else ""
        return (
            f"No spending-over-time data found for {scope} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}."
            f"{no_results_note}"
        )

    lines = [
        f"{_format_time_period(r.time_period)}: ${r.aggregated_amount:,.2f}"
        for r in response.results
    ]
    note = _format_api_messages(response.messages)
    if naics_note:
        note += f"\n\n({naics_note})"
    return _wrap_untrusted("\n".join(lines) + note)

"""get_spending_by_geography - spending broken down by state/county/district/etc."""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    GeographyTypeResult,
    SpendingByGeographyResponse,
    USASpendingAPIError,
)

from ...recipient_types import RecipientType
from ...response_shaping import year_label
from ...singletons import _get_usaspending_client
from ...tool_filters import (
    AwardType,
    DateType,
    GeoLayer,
    GeoScope,
    Scope,
    _build_filters,
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

# Self-imposed, not API-driven - spending_by_geography has no limit/page
# param and returns every matching region (e.g. ~3,100+ for an unfiltered
# US county breakdown). Caps what reaches the model per call.
MAX_GEOGRAPHY_RESULTS = 20


@traceable(run_type="tool", name="get_spending_by_geography_raw")
def get_spending_by_geography_raw(
    scope: GeoScope,
    geo_layer: GeoLayer,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    agency_name: str | None = None,
    geo_layer_filters: list[str] | None = None,
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
) -> SpendingByGeographyResponse:
    client = _get_usaspending_client()
    filters = _build_filters(
        client, agency_name, time_period_type, start_year, end_year,
        award_type=award_type, recipient_name=recipient_name, recipient_id=recipient_id,
        min_amount=min_amount, max_amount=max_amount,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        keywords=keywords, date_type=date_type,
        place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
        award_id=award_id, recipient_type=recipient_type, description=description,
    )
    return client.spending_by_geography(filters, scope, geo_layer, geo_layer_filters)


def _format_geography_result(result: GeographyTypeResult) -> str:
    name = result.display_name or result.shape_code or "Unmapped/unknown location"
    line = f"{name}: ${result.aggregated_amount:,.2f}"
    if result.per_capita is not None:
        line += f" (${result.per_capita:,.2f} per capita, population {result.population:,})"
    return line


@beta_tool
def get_spending_by_geography(
    scope: GeoScope,
    geo_layer: GeoLayer,
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    agency_name: str | None = None,
    geo_layer_filters: list[str] | None = None,
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
) -> str:
    """Get spending broken down by geographic region (state, county, congressional district, or country) for a fiscal year range, ranked by total amount descending. Use this for "which states/counties/districts/countries got the most X funding" — a full breakdown in one call, instead of checking one place at a time.

    At least one of agency_name, recipient_name, recipient_id, performed_in_state, recipient_in_state,
    naics_code, psc_code, cfda_program, or keywords must be given - a query scoped by none of them
    would mean all federal spending, ever, which this tool refuses rather than silently running.

    Population and per-capita spending are computed by the live API itself and included automatically
    when available. Both reflect current data, not the population/geography at the time of the
    queried period - do not present a per-capita or district figure as historically precise for an
    older fiscal year.

    Args:
        scope: "place_of_performance" (where the work was done) or "recipient_location" (where the
            recipient is based) - these can differ substantially for the same query.
        geo_layer: "state", "county", "district" (congressional district), or "country" - the
            granularity to break results down by.
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 or CY2021. Data is only available from FY2008 (or CY2007)
            onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
        geo_layer_filters: Optional. Restrict results to specific regions only - state codes (e.g.
            ["VA", "MD"]), county FIPS codes, congressional district codes, or ISO 3166-1 alpha-3
            country codes, matching geo_layer. Omit to get every region.
        award_type: Optional. Restrict to one award type or bucket. Omit to include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name contains
            this text, e.g. "Leidos". An approximate text match - prefer recipient_id when available.
        recipient_id: Optional. An exact recipient_id from a prior search_recipients or
            get_recipient_details call. Prefer this over recipient_name for precision.
        min_amount: Optional. Restrict to spending of at least this dollar amount.
        max_amount: Optional. Restrict to spending of at most this dollar amount.
        performed_in_state: Optional. Restrict to spending on work performed in this US state -
            combining this with geo_layer="county" gives a breakdown of counties within one state.
        recipient_in_state: Optional. Restrict to spending on recipients headquartered in this
            US state.
        performed_in_county: Optional. A specific county where work was performed - a 3-digit
            FIPS code (e.g. "025" for Yavapai County, AZ), not a name. Requires
            performed_in_state also be set. Use resolve_county_fips to find the code from a
            county name - do not guess or construct one.
        recipient_in_county: Optional. Same as performed_in_county, but for the recipient's
            location. Requires recipient_in_state also be set.
        performed_in_city: Optional. Restrict to work performed in this city, e.g. "Livermore".
        recipient_in_city: Optional. Same as performed_in_city, but for the recipient's location.
        performed_in_zip: Optional. Restrict to work performed in this 5-digit zip code.
        recipient_in_zip: Optional. Restrict to a recipient located in this 5-digit zip code.
        performed_in_district: Optional. A specific congressional district where work was
            performed - a 2-digit number (e.g. "01"), paired with performed_in_state.
        recipient_in_district: Optional. Same as performed_in_district, but for the recipient's
            location, paired with recipient_in_state.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
            Do NOT restate the award_type/category itself here (e.g. "grant", "contracts",
            "cooperative agreement") - award_type already scopes that precisely, and doing so
            on top silently narrows results to only the awards whose description text happens
            to contain that literal word (most awards of that type don't say it) rather than
            broadening or duplicating the filter. Rejected with an error if given alone.
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default), date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code.
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number, format NN.NNN.
        award_id: Optional. Restrict to a single known award by its plain Award ID (PIID/FAIN/URI) -
            a fuzzy text match, not an exact-id lookup.
        recipient_type: Optional. Restrict to recipients tagged with this business/recipient
            type, e.g. "small_business", "woman_owned_business", "nonprofit", "higher_education".
            Not sufficient scope on its own (like award_type).
        description: Optional. Restrict to awards whose own description text matches this
            phrase, e.g. "vaccine research". Distinct from keywords, which also matches other
            text fields (recipient name, PIID/FAIN/URI, NAICS/PSC description).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    scope_label_str = _scope_label(
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
        response = get_spending_by_geography_raw(
            scope, geo_layer, time_period_type, start_year, end_year,
            agency_name=agency_name, geo_layer_filters=geo_layer_filters,
            award_type=award_type, recipient_name=recipient_name, recipient_id=recipient_id,
            min_amount=min_amount, max_amount=max_amount,
            performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
            performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
            performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
            performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
            performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
            keywords=keywords, date_type=date_type,
            place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
            naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
            award_id=award_id, recipient_type=recipient_type, description=description,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_geography failed for %s: %s", scope_label_str, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "scope": scope, "geo_layer": geo_layer,
            "start_year": start_year, "end_year": end_year, "time_period_type": time_period_type,
        },
        agency_name=agency_name, award_type=award_type, recipient_name=recipient_name,
        recipient_id=recipient_id, min_amount=min_amount, max_amount=max_amount,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        keywords=keywords, date_type=date_type,
        place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
        award_id=award_id, recipient_type=recipient_type, description=description,
    )
    _record_tool_call("get_spending_by_geography", response, context)

    if not response.results:
        no_results = (
            f"No spending-by-geography data found for {scope_label_str} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}."
        )
        if geo_layer in ("county", "district"):
            # county/district geocoding can lag/be incomplete even when state-level data exists.
            no_results += (" This does not necessarily mean there was no spending - "
                            "try geo_layer=\"state\" or search_awards to confirm before concluding there was none.")
        return no_results

    results_sorted = sorted(response.results, key=lambda r: -r.aggregated_amount)
    shown = results_sorted[:MAX_GEOGRAPHY_RESULTS]
    lines = [_format_geography_result(r) for r in shown]

    note_parts = []
    if len(results_sorted) > MAX_GEOGRAPHY_RESULTS:
        note_parts.append(
            f"this shows the top {MAX_GEOGRAPHY_RESULTS} of {len(results_sorted)} regions by amount - "
            "do not present this as the complete list"
        )
    if any(r.per_capita is not None for r in shown):
        note_parts.append(
            "population/per-capita figures reflect current data, not the period being queried"
        )
    if geo_layer == "district":
        note_parts.append("congressional district boundaries reflect the current map, not the "
                           "boundaries in effect during the queried period")
    note = f"\n\n(Note: {'; '.join(note_parts)}.)" if note_parts else ""

    return _wrap_untrusted("\n".join(lines) + note + _format_api_messages(response.messages))

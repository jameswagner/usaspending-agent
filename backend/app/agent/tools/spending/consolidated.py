"""Spike (#265): two candidate consolidated shapes for the six
filter-sharing spending tools (search_awards, search_subawards,
search_transactions, get_spending_by_category, get_spending_over_time,
get_spending_by_geography), for real bind_tools()-path token measurement
and tool-selection-eval comparison against the current six-tool baseline.

NOT registered anywhere live - not imported by langgraph_tools.py's
_BETA_TOOLS, not reachable from ask(). See docs/spike-265-consolidate-tools.md
for the go/no-go writeup.

Both shapes delegate to the six existing @beta_tool functions rather than
re-deriving their _raw-call/_build_filters/formatting/_record_tool_call
logic - this file owns only the routing decision the spike measures
(level/group_by -> the right underlying tool), not a second copy of
everything downstream of that decision.
"""
from __future__ import annotations

from typing import Literal

from anthropic import beta_tool

from ...contract_type_codes import ContractPricingType, ExtentCompetedType, SetAsideType
from ...recipient_types import RecipientType
from ...tool_filters import AwardType, DateType, GeoLayer, GeoScope, Scope, SortBy
from .category import VALID_CATEGORIES, get_spending_by_category
from .geography import get_spending_by_geography
from .over_time import Group, get_spending_over_time
from .search import search_awards, search_subawards, search_transactions

Level = Literal["award", "subaward", "transaction"]
GroupBy = Literal[
    "awarding_agency", "awarding_subagency", "cfda", "country", "county",
    "defc", "district", "federal_account", "funding_agency",
    "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
    "state_territory", "time", "geography",
]

assert set(GroupBy.__args__) - {"time", "geography"} == VALID_CATEGORIES


def _route_records(
    level: Level,
    *,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    award_type: AwardType,
    limit: int,
    sort_by: SortBy,
    agency_name: str | None,
    recipient_name: str | None,
    min_amount: float | None,
    max_amount: float | None,
    performed_in_state: str | None,
    recipient_in_state: str | None,
    performed_in_county: str | None,
    recipient_in_county: str | None,
    performed_in_city: str | None,
    recipient_in_city: str | None,
    performed_in_zip: str | None,
    recipient_in_zip: str | None,
    performed_in_district: str | None,
    recipient_in_district: str | None,
    keywords: str | None,
    date_type: DateType | None,
    place_of_performance_scope: Scope | None,
    recipient_scope: Scope | None,
    naics_code: str | None,
    psc_code: str | None,
    cfda_program: str | None,
    award_id: str | None,
    recipient_type: RecipientType | None,
    description: str | None,
    tas_code: str | None,
    federal_account: str | None,
    def_codes: list[str] | None,
    contract_pricing_type: list[ContractPricingType] | None,
    set_aside_type: list[SetAsideType] | None,
    extent_competed_type: list[ExtentCompetedType] | None,
) -> str:
    shared = {
        "time_period_type": time_period_type, "start_year": start_year, "end_year": end_year,
        "award_type": award_type, "agency_name": agency_name,
        "min_amount": min_amount, "max_amount": max_amount,
        "performed_in_state": performed_in_state, "performed_in_county": performed_in_county,
        "performed_in_city": performed_in_city, "performed_in_zip": performed_in_zip,
        "performed_in_district": performed_in_district,
        "keywords": keywords, "date_type": date_type,
        "place_of_performance_scope": place_of_performance_scope, "recipient_scope": recipient_scope,
        "naics_code": naics_code, "psc_code": psc_code, "cfda_program": cfda_program,
        "award_id": award_id, "recipient_type": recipient_type, "description": description,
    }
    if level == "subaward":
        return search_subawards(
            **shared,
            limit=limit,
            subrecipient_name=recipient_name,
            subrecipient_in_state=recipient_in_state,
            subrecipient_in_county=recipient_in_county,
            subrecipient_in_city=recipient_in_city,
            subrecipient_in_zip=recipient_in_zip,
            subrecipient_in_district=recipient_in_district,
        )
    if level == "transaction":
        return search_transactions(
            **shared,
            limit=limit,
            sort_by=sort_by if sort_by in ("amount", "recency") else "amount",
            recipient_name=recipient_name,
            recipient_in_state=recipient_in_state, recipient_in_county=recipient_in_county,
            recipient_in_city=recipient_in_city, recipient_in_zip=recipient_in_zip,
            recipient_in_district=recipient_in_district,
            tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
        )
    return search_awards(
        **shared,
        limit=limit, sort_by=sort_by,
        recipient_name=recipient_name,
        recipient_in_state=recipient_in_state, recipient_in_county=recipient_in_county,
        recipient_in_city=recipient_in_city, recipient_in_zip=recipient_in_zip,
        recipient_in_district=recipient_in_district,
        tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
        contract_pricing_type=contract_pricing_type, set_aside_type=set_aside_type,
        extent_competed_type=extent_competed_type,
    )


def _route_aggregate(
    group_by: GroupBy,
    *,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    time_grouping: Group,
    scope: GeoScope | None,
    geo_layer: GeoLayer | None,
    geo_layer_filters: list[str] | None,
    award_type: AwardType | None,
    limit: int,
    agency_name: str | None,
    recipient_name: str | None,
    recipient_id: str | None,
    min_amount: float | None,
    max_amount: float | None,
    performed_in_state: str | None,
    recipient_in_state: str | None,
    performed_in_county: str | None,
    recipient_in_county: str | None,
    performed_in_city: str | None,
    recipient_in_city: str | None,
    performed_in_zip: str | None,
    recipient_in_zip: str | None,
    performed_in_district: str | None,
    recipient_in_district: str | None,
    keywords: str | None,
    date_type: DateType | None,
    place_of_performance_scope: Scope | None,
    recipient_scope: Scope | None,
    naics_code: str | None,
    psc_code: str | None,
    cfda_program: str | None,
    award_id: str | None,
    recipient_type: RecipientType | None,
    description: str | None,
    def_codes: list[str] | None,
) -> str:
    shared = {
        "time_period_type": time_period_type, "start_year": start_year, "end_year": end_year,
        "award_type": award_type, "agency_name": agency_name,
        "recipient_name": recipient_name, "recipient_id": recipient_id,
        "min_amount": min_amount, "max_amount": max_amount,
        "performed_in_state": performed_in_state, "recipient_in_state": recipient_in_state,
        "performed_in_county": performed_in_county, "recipient_in_county": recipient_in_county,
        "performed_in_city": performed_in_city, "recipient_in_city": recipient_in_city,
        "performed_in_zip": performed_in_zip, "recipient_in_zip": recipient_in_zip,
        "performed_in_district": performed_in_district, "recipient_in_district": recipient_in_district,
        "keywords": keywords, "date_type": date_type,
        "place_of_performance_scope": place_of_performance_scope, "recipient_scope": recipient_scope,
        "naics_code": naics_code, "psc_code": psc_code, "cfda_program": cfda_program,
        "award_id": award_id, "recipient_type": recipient_type, "description": description,
    }
    if group_by == "time":
        return get_spending_over_time(**shared, group=time_grouping, def_codes=def_codes)
    if group_by == "geography":
        if scope is None or geo_layer is None:
            return "This query failed: group_by='geography' requires scope and geo_layer."
        return get_spending_by_geography(
            **shared, scope=scope, geo_layer=geo_layer, geo_layer_filters=geo_layer_filters,
        )
    return get_spending_by_category(category=group_by, **shared, limit=limit, def_codes=def_codes)


# ---------------------------------------------------------------------------
# Shape A: one tool, `level` (rows) xor `group_by` (aggregate) discriminates
# everything else.
# ---------------------------------------------------------------------------


@beta_tool
def query_spending(
    level: Level,
    *,
    group_by: GroupBy | None = None,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    sort_by: SortBy = "amount",
    time_grouping: Group = "fiscal_year",
    scope: GeoScope | None = None,
    geo_layer: GeoLayer | None = None,
    geo_layer_filters: list[str] | None = None,
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
    tas_code: str | None = None,
    federal_account: str | None = None,
    def_codes: list[str] | None = None,
    contract_pricing_type: list[ContractPricingType] | None = None,
    set_aside_type: list[SetAsideType] | None = None,
    extent_competed_type: list[ExtentCompetedType] | None = None,
) -> str:
    """Query federal spending: either individual records at a given level, or an
    aggregate rollup. Omit group_by for individual ranked award/subaward/transaction
    rows (set level accordingly) - use this for "show me X's awards/contracts" or
    "who received money from X" questions. Set group_by for a summed breakdown
    instead: a category dimension (naics/psc/cfda/awarding_agency/recipient/
    state_territory/...), "time" for a period-over-period trend, or "geography"
    for a place-of-performance/recipient-location breakdown - use this for
    "how is X's spending broken down by Y" or "how has X's spending trended"
    questions, as opposed to a ranked list of individual records.

    Args:
        level: "award" for prime awards (default choice for records), "subaward"
            for subawards (recipient_/location_ fields below then filter the
            SUB-recipient, not the prime - opposite of normal), "transaction" for
            individual transactions/modifications rather than award-level totals.
            Still required even when group_by is set, to pick which underlying
            record type the aggregate is computed over.
        group_by: Optional. Omit to return individual ranked rows for the chosen
            level (sort_by controls ranking). Set to a category name (naics, psc,
            cfda, awarding_agency, awarding_subagency, funding_agency,
            funding_subagency, recipient, recipient_duns, state_territory, county,
            district, country, federal_account, defc) for a summed breakdown by
            that dimension, "time" for a period-over-period trend (time_grouping
            controls the period size), or "geography" for a place-of-performance/
            recipient-location breakdown (scope and geo_layer become required).
        time_period_type: "fiscal" (default) or "calendar".
        start_year: First year to include.
        end_year: Last year to include.
        award_type: contracts/grants/loans or a specific sub-type (default contracts).
        limit: Max rows/categories to return (default 5). Ignored for group_by="time".
        sort_by: Ranking field for record rows only (amount default, outlays, subsidy_cost,
            recency). Ignored when group_by is set.
        time_grouping: Period size when group_by="time" - fiscal_year (default),
            calendar_year, quarter, month. Ignored otherwise.
        scope: Required when group_by="geography" - "place_of_performance" or "recipient_location".
        geo_layer: Required when group_by="geography" - "state", "county", "district", or "country".
        geo_layer_filters: Optional, only used when group_by="geography" - restrict to these
            specific state/country codes.
        agency_name: Optional. The awarding agency's name.
        recipient_name: Optional. Recipient name text match (the sub-recipient's, if level="subaward").
        recipient_id: Optional. Exact recipient id - only honored for group_by aggregates, not records.
        min_amount: Optional. Minimum award/transaction amount.
        max_amount: Optional. Maximum award/transaction amount.
        performed_in_state: Optional. Where the work was performed.
        recipient_in_state: Optional. Where the recipient (or sub-recipient) is located.
        performed_in_county: Optional. 3-digit FIPS code, requires performed_in_state.
        recipient_in_county: Optional. 3-digit FIPS code, requires recipient_in_state.
        performed_in_city: Optional.
        recipient_in_city: Optional.
        performed_in_zip: Optional.
        recipient_in_zip: Optional.
        performed_in_district: Optional. 2-digit number, requires performed_in_state.
        recipient_in_district: Optional. 2-digit number, requires recipient_in_state.
        keywords: Optional. Free-text search over descriptions - don't restate award_type.
        date_type: Optional. action_date (default), date_signed, last_modified_date, new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign".
        recipient_scope: Optional. "domestic" or "foreign".
        naics_code: Optional. Exact NAICS code.
        psc_code: Optional. Exact 4-character PSC.
        cfda_program: Optional. Exact CFDA number, NN.NNN.
        award_id: Optional. Fuzzy PIID/FAIN/URI match, records only.
        recipient_type: Optional. Business/recipient type tag.
        description: Optional. Award description text match.
        tas_code: Optional. Exact Treasury Account Symbol, records only (award/transaction levels).
        federal_account: Optional. Exact federal account, records only.
        def_codes: Optional. Disaster Emergency Fund Codes, or group aliases like "covid"/"infrastructure".
        contract_pricing_type: Optional. Records only, level="award". Contract-only.
        set_aside_type: Optional. Records only, level="award". Contract-only.
        extent_competed_type: Optional. Records only, level="award". Contract-only.
    """
    if group_by is None:
        return _route_records(
            level,
            time_period_type=time_period_type, start_year=start_year, end_year=end_year,
            award_type=award_type, limit=limit, sort_by=sort_by, agency_name=agency_name,
            recipient_name=recipient_name, min_amount=min_amount, max_amount=max_amount,
            performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
            performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
            performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
            performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
            performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
            keywords=keywords, date_type=date_type,
            place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
            naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
            award_id=award_id, recipient_type=recipient_type, description=description,
            tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
            contract_pricing_type=contract_pricing_type, set_aside_type=set_aside_type,
            extent_competed_type=extent_competed_type,
        )
    return _route_aggregate(
        group_by,
        time_period_type=time_period_type, start_year=start_year, end_year=end_year,
        time_grouping=time_grouping, scope=scope, geo_layer=geo_layer,
        geo_layer_filters=geo_layer_filters, award_type=award_type, limit=limit,
        agency_name=agency_name, recipient_name=recipient_name, recipient_id=recipient_id,
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
        def_codes=def_codes,
    )


# ---------------------------------------------------------------------------
# Shape B: two tools - rows vs. an aggregate rollup as a tool-name-level choice.
# ---------------------------------------------------------------------------


@beta_tool
def search_spending_records(
    level: Level,
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    sort_by: SortBy = "amount",
    agency_name: str | None = None,
    recipient_name: str | None = None,
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
    tas_code: str | None = None,
    federal_account: str | None = None,
    def_codes: list[str] | None = None,
    contract_pricing_type: list[ContractPricingType] | None = None,
    set_aside_type: list[SetAsideType] | None = None,
    extent_competed_type: list[ExtentCompetedType] | None = None,
) -> str:
    """Search for individual award, subaward, or transaction records matching
    filters. Returns ranked/paginated rows, not a total - use this for
    "show me X's awards/contracts" or "who received money from X" questions,
    as opposed to aggregate_spending's summed breakdown/trend.

    Args:
        level: "award" for prime awards, "subaward" for subawards (recipient_/
            location_ fields then filter the SUB-recipient, not the prime),
            "transaction" for individual transactions/modifications.
        time_period_type: "fiscal" (default) or "calendar".
        start_year: First year to include.
        end_year: Last year to include.
        award_type: contracts/grants/loans or a specific sub-type (default contracts).
        limit: Max results to return (default 5).
        sort_by: amount (default), outlays, subsidy_cost, recency.
        agency_name: Optional. The awarding agency's name.
        recipient_name: Optional. Recipient (or sub-recipient, if level="subaward") name text match.
        min_amount: Optional.
        max_amount: Optional.
        performed_in_state: Optional. Where the work was performed.
        recipient_in_state: Optional. Where the recipient is located.
        performed_in_county: Optional. 3-digit FIPS, requires performed_in_state.
        recipient_in_county: Optional. 3-digit FIPS, requires recipient_in_state.
        performed_in_city: Optional.
        recipient_in_city: Optional.
        performed_in_zip: Optional.
        recipient_in_zip: Optional.
        performed_in_district: Optional. 2-digit number, requires performed_in_state.
        recipient_in_district: Optional. 2-digit number, requires recipient_in_state.
        keywords: Optional. Free-text over descriptions - don't restate award_type.
        date_type: Optional. action_date (default), date_signed, last_modified_date, new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign".
        recipient_scope: Optional. "domestic" or "foreign".
        naics_code: Optional. Exact NAICS code.
        psc_code: Optional. Exact 4-character PSC.
        cfda_program: Optional. Exact CFDA number, NN.NNN.
        award_id: Optional. Fuzzy PIID/FAIN/URI match.
        recipient_type: Optional. Business/recipient type tag.
        description: Optional. Award description text match.
        tas_code: Optional. Exact Treasury Account Symbol, award/transaction levels only.
        federal_account: Optional. Exact federal account, award/transaction levels only.
        def_codes: Optional. Disaster Emergency Fund Codes, or group aliases like "covid".
        contract_pricing_type: Optional. level="award" only. Contract-only.
        set_aside_type: Optional. level="award" only. Contract-only.
        extent_competed_type: Optional. level="award" only. Contract-only.
    """
    return _route_records(
        level,
        time_period_type=time_period_type, start_year=start_year, end_year=end_year,
        award_type=award_type, limit=limit, sort_by=sort_by, agency_name=agency_name,
        recipient_name=recipient_name, min_amount=min_amount, max_amount=max_amount,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        keywords=keywords, date_type=date_type,
        place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
        award_id=award_id, recipient_type=recipient_type, description=description,
        tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
        contract_pricing_type=contract_pricing_type, set_aside_type=set_aside_type,
        extent_competed_type=extent_competed_type,
    )


@beta_tool
def aggregate_spending(
    group_by: GroupBy,
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    award_type: AwardType | None = None,
    limit: int = 5,
    time_grouping: Group = "fiscal_year",
    scope: GeoScope | None = None,
    geo_layer: GeoLayer | None = None,
    geo_layer_filters: list[str] | None = None,
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
    def_codes: list[str] | None = None,
) -> str:
    """Sum federal spending grouped by a category, over time, or by geography.
    Returns a rollup, not individual records - use this for "how is X's
    spending broken down by Y" or "how has X's spending trended" questions, as
    opposed to search_spending_records's ranked list of individual rows.

    Args:
        group_by: A category dimension (naics, psc, cfda, awarding_agency,
            awarding_subagency, funding_agency, funding_subagency, recipient,
            recipient_duns, state_territory, county, district, country,
            federal_account, defc), "time" for a period-over-period trend
            (time_grouping controls the period size), or "geography" for a
            place-of-performance/recipient-location breakdown (scope and
            geo_layer become required).
        time_period_type: "fiscal" (default) or "calendar".
        start_year: First year to include.
        end_year: Last year to include.
        award_type: Optional filter to contracts/grants/loans or a sub-type.
        limit: Max categories to return (default 5). Ignored for group_by="time".
        time_grouping: Period size when group_by="time" - fiscal_year (default),
            calendar_year, quarter, month. Ignored otherwise.
        scope: Required when group_by="geography" - "place_of_performance" or "recipient_location".
        geo_layer: Required when group_by="geography" - "state", "county", "district", or "country".
        geo_layer_filters: Optional, only used when group_by="geography".
        agency_name: Optional. The awarding agency's name - omit for a cross-agency,
            recipient-only question, but then recipient_name/recipient_id must be set.
        recipient_name: Optional. Recipient name text match.
        recipient_id: Optional. Exact recipient id, more precise than recipient_name.
        min_amount: Optional.
        max_amount: Optional.
        performed_in_state: Optional.
        recipient_in_state: Optional.
        performed_in_county: Optional. 3-digit FIPS, requires performed_in_state.
        recipient_in_county: Optional. 3-digit FIPS, requires recipient_in_state.
        performed_in_city: Optional.
        recipient_in_city: Optional.
        performed_in_zip: Optional.
        recipient_in_zip: Optional.
        performed_in_district: Optional. 2-digit number, requires performed_in_state.
        recipient_in_district: Optional. 2-digit number, requires recipient_in_state.
        keywords: Optional. Free-text over descriptions.
        date_type: Optional. action_date (default), date_signed, last_modified_date, new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign".
        recipient_scope: Optional. "domestic" or "foreign".
        naics_code: Optional. Exact NAICS code.
        psc_code: Optional. Exact 4-character PSC.
        cfda_program: Optional. Exact CFDA number, NN.NNN.
        award_id: Optional.
        recipient_type: Optional. Business/recipient type tag.
        description: Optional.
        def_codes: Optional. Disaster Emergency Fund Codes, or group aliases like "covid".
    """
    return _route_aggregate(
        group_by,
        time_period_type=time_period_type, start_year=start_year, end_year=end_year,
        time_grouping=time_grouping, scope=scope, geo_layer=geo_layer,
        geo_layer_filters=geo_layer_filters, award_type=award_type, limit=limit,
        agency_name=agency_name, recipient_name=recipient_name, recipient_id=recipient_id,
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
        def_codes=def_codes,
    )

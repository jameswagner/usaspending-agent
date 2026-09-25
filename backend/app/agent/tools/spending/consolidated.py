"""query_spending - the six former filter-sharing spending tools
(search_awards, search_subawards, search_transactions,
get_spending_by_category, get_spending_over_time,
get_spending_by_geography) consolidated into one. level
(award/subaward/transaction) picks individual records; group_by picks an
aggregate rollup instead.

Delegates to the six original @beta_tool functions rather than
re-deriving their _raw-call/_build_filters/formatting/_record_tool_call
logic - this file owns only the level/group_by routing decision, not a
second copy of everything downstream of it.
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
# state_territory/county/district/country omitted: always redundant with
# group_by="geography"+geo_layer, which is also strictly more capable (recipient_location scope).
SpendingGroupBy = Literal[
    "awarding_agency", "awarding_subagency", "cfda", "defc", "federal_account",
    "funding_agency", "funding_subagency", "naics", "psc", "recipient",
    "recipient_duns", "time", "geography",
]

_LOCATION_CATEGORIES = {"state_territory", "county", "district", "country"}
assert set(SpendingGroupBy.__args__) - {"time", "geography"} == VALID_CATEGORIES - _LOCATION_CATEGORIES


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


_SPENDING_LEVEL_OF_LEVEL: dict[Level, str] = {
    "award": "transactions", "subaward": "subawards", "transaction": "transactions",
}


def _route_aggregate(
    group_by: SpendingGroupBy,
    *,
    level: Level,
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
    spending_level = _SPENDING_LEVEL_OF_LEVEL[level]
    # Both aggregate endpoints reject recipient_id outright for subaward queries - fail fast, not via a live 400.
    if spending_level == "subawards" and recipient_id is not None:
        return (
            "This query failed: recipient_id is not supported for subaward "
            "queries (a live API restriction, not a bug) - use recipient_name instead."
        )
    if group_by == "time":
        return get_spending_over_time(**shared, group=time_grouping, def_codes=def_codes, spending_level=spending_level)
    if group_by == "geography":
        if scope is None or geo_layer is None:
            return "This query failed: group_by='geography' requires scope and geo_layer."
        # spending_by_geography has no spending_level - hardcoded to "transactions" by
        # design (client.spending_by_geography's own docstring: other modes overcounted
        # 60%+ in testing), so level has no effect here.
        return get_spending_by_geography(
            **shared, scope=scope, geo_layer=geo_layer, geo_layer_filters=geo_layer_filters,
        )
    return get_spending_by_category(
        category=group_by, **shared, limit=limit, def_codes=def_codes, spending_level=spending_level,
    )


@beta_tool
def query_spending(
    level: Level,
    *,
    group_by: SpendingGroupBy | None = None,
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
    instead: a category dimension (naics/psc/cfda/awarding_agency/recipient/...),
    "time" for a period-over-period trend, or "geography" for a
    place-of-performance/recipient-location/state/county/district/country
    breakdown - use this for "how is X's spending broken down by Y" or "how has
    X's spending trended" questions, as opposed to a ranked list of individual
    records.

    IMPORTANT about fiscal-year scoping: by default (date_type omitted), an award/
    transaction/subaward appears here if it had ANY activity in the queried fiscal
    year - not "this dollar amount was specifically obligated in this year." A
    multi-year award active in more than one fiscal year appears in results for
    EACH of those years, and group_by=None row totals are each award's current
    LIFETIME value, not a per-year figure - do not sum these across multiple
    fiscal-year calls as if period-scoped and additive; use group_by="time" for a
    genuinely period-scoped, non-duplicative total instead. A surprising $0 or
    empty result under the default date_type is often correct, not a failure -
    explain it (no NEW activity that year) rather than retrying with different
    filters. If the question is really "what NEW awards did X get in FY2024"
    rather than "what was X active on," set date_type="new_awards_only" instead.

    Aggregates (group_by set to anything) and level="subaward" records both
    require real scope (agency_name, recipient_name, a location, naics/psc/cfda,
    or similar, depending on branch) - a query calling this without any of them
    is refused with a clear error naming what's missing; level="award"/"transaction"
    records need no scope beyond award_type and the fiscal year range.

    When comparing the same entity across more than one call (e.g. prime vs.
    subaward totals, or two time periods, or two agencies), reuse the exact same
    recipient scoping in every call - once recipient_id is resolved for an entity,
    use it every time you scope by that entity again, never switching to
    recipient_name/recipient_search_text partway through a comparison, since the two
    are not guaranteed to match the same set of records. The one exception:
    level="subaward" never accepts recipient_id at all (see recipient_id below) -
    use recipient_name for every subaward-level call about that entity instead,
    consistently, not just as a fallback after recipient_id fails - and prefer a
    resolved uei/duns there over a bare name (see recipient_name below).

    Args:
        level: "award" for prime awards (default choice for records), "subaward"
            for subawards (recipient_/location_ fields below then filter the
            SUB-recipient, not the prime - opposite of normal), "transaction" for
            individual transactions/modifications rather than award-level totals -
            one multi-year award with 10 mods is ONE row at level="award" but up to
            10 separate rows at level="transaction", each with its own action date
            and amount.
            Still required even when group_by is set: level="subaward" with
            group_by="time" or a category name gives a real subaward-dollar
            total/trend (not individual subaward records - use group_by=None for
            those instead). level="award" vs "transaction" makes no difference to
            an aggregate - both compute the same period-scoped grand total; that
            distinction only matters for individual rows (group_by=None). Has no
            effect on group_by="geography", which only ever aggregates prime
            transactions.
        group_by: Optional. Omit to return individual ranked rows for the chosen
            level (sort_by controls ranking). Set to a category name (naics, psc,
            cfda, awarding_agency, awarding_subagency, funding_agency,
            funding_subagency, recipient, recipient_duns, federal_account, defc)
            for a summed breakdown by that dimension, "time" for a
            period-over-period trend (time_grouping controls the period size), or
            "geography" for a state/county/district/country breakdown by
            place-of-performance or recipient-location (scope and geo_layer become
            required) - "geography" is also the only way to break down by
            state/county/district/country at all, there is no separate category
            value for those.
        time_period_type: "fiscal" (default) or "calendar".
        start_year: First year to include.
        end_year: Last year to include.
        award_type: contracts/grants/loans or a specific sub-type (default contracts).
        limit: Max rows/categories to return (default 5). Ignored for group_by="time" and
            group_by="geography" (always capped at 20 regions internally, not this value).
        sort_by: Ranking field for record rows only (amount default, outlays, subsidy_cost,
            recency). Ignored when group_by is set.
        time_grouping: Period size when group_by="time" - fiscal_year (default),
            calendar_year, quarter, month. Ignored otherwise.
        scope: Required when group_by="geography" - "place_of_performance" or "recipient_location".
        geo_layer: Required when group_by="geography" - "state", "county", "district", or "country".
        geo_layer_filters: Optional, only used when group_by="geography" - restrict to these
            specific state/country codes.
        agency_name: Optional. The awarding agency's name.
        recipient_name: Optional. Recipient name text match (the sub-recipient's, if level="subaward")
            - a bare company name can sweep in every distinct entity sharing that name (confirmed
            live: "Boeing" alone returned 66x more than the one specific Boeing entity). When
            level="subaward" and you already have a uei or duns for the entity (e.g. from
            search_recipients/get_recipient_details), pass THAT here instead of the bare name -
            it's still a text match, but a uei/duns has no other entity to collide with, giving
            the same effective precision recipient_id gives non-subaward calls.
        recipient_id: Optional. Exact recipient id - only honored for group_by aggregates, not
            records, and NOT supported at all when level="subaward" (a live API restriction -
            use recipient_name there instead, every time, not just after this fails once).
            Otherwise prefer this over recipient_name whenever you have it (e.g. from
            search_recipients), and once you've used it for an entity, keep using it for every
            later call about that same entity in this answer.
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
        naics_code: Optional. Exact NAICS code - or a plain-English industry description (see
            resolve_naics_code above); if ambiguous, use group_by="naics" to browse instead of guessing.
        psc_code: Optional. Exact 4-character PSC. Same guidance as naics_code: use
            group_by="psc" to browse if you don't have the exact code.
        cfda_program: Optional. Exact CFDA number, NN.NNN.
        award_id: Optional. Fuzzy PIID/FAIN/URI match, records only. For an IDV's own PIID
            (a contract vehicle, not a plain contract), pass award_type="idv" too - IDV codes
            are disjoint from A/B/C/D, so the default award_type="contracts" won't find it.
        recipient_type: Optional. Business/recipient type tag - sufficient scope on its own for
            aggregates, unlike award_type. NOT reversed for level="subaward" (unlike
            recipient_name/recipient_in_*/description above/below) - still describes the PRIME
            recipient's business type even when scoping a subaward-level call.
        description: Optional. Award (or, if level="subaward", the sub-award's own) description
            text match - distinct from keywords, which also matches recipient name/PIID/FAIN/NAICS/PSC.
        tas_code: Optional. Exact Treasury Account Symbol, records only (award/transaction levels).
            Must be a real code - get it from a get_award_funding_breakdown call, not guessed.
        federal_account: Optional. Exact federal account (the AID-MAIN pair one level up from a
            full TAS), records only - the value shown on a get_award_funding_breakdown row.
        def_codes: Optional. Disaster Emergency Fund Codes, or group aliases like "covid"/"infrastructure".
            For a spending-by-DEFC breakdown instead of filtering to a specific one, use group_by="defc".
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
        level=level,
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

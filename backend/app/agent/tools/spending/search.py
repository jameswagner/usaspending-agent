"""search_awards / search_subawards / search_transactions - ranked, paginated
award, subaward, and transaction lists, kept in one module since they share
the same date-range/award-type shape. Named search.py, not awards.py, to
avoid colliding with the existing tools/awards.py (award-detail/IDV lookups -
a different concern).
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending import SearchAwardsResponse, USASpendingAPIError

from ...recipient_types import RecipientType
from ...response_shaping import year_label, year_range_to_date_range
from ...singletons import _get_usaspending_client
from ...tool_filters import (
    DISASTER_BREAKOUT_FIELDS,
    SEARCH_AWARDS_FIELDS_BASE,
    SUBAWARD_FIELDS,
    TRANSACTION_FIELDS_BASE,
    AwardType,
    DateType,
    Scope,
    SortBy,
    TransactionSortBy,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
    _other_award_type_categories_to_try,
    _record_optional_filter_context,
    _sort_field_for_award_type,
    _transaction_amount_field_for_award_type,
)
from .._shared import (
    _check_tool_call_budget,
    _format_api_messages,
    _record_tool_call,
    _scope_label,
    _truncation_note,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="search_awards_raw")
def search_awards_raw(
    agency_name: str | None,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    sort_by: SortBy = "amount",
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
) -> SearchAwardsResponse:
    """Call the API once, return the structured response (results +
    page_metadata), sorted largest-first by sort_by (default "amount":
    Award Amount, or Loan Value for loan award types - see
    _amount_field_for_award_type; see _sort_field_for_award_type for the
    other sort_by options). The live API's own default order is
    essentially arbitrary - an unsorted "top 5" NSF FY2023 contracts
    query once returned awards from $7K to $7.2M while the true largest
    that year ($3.13B) never appeared. Unlike the filter params, which
    are optional and behavior-preserving when omitted, this sort is NOT
    optional - there's no meaningful default ordering to preserve.

    Filter resolution (agency, award_type, recipient, amount, location) is
    delegated to _build_filters, same as the other two spending tools. No
    recipient_id param here, unlike the other two: this endpoint silently
    ignores that filter entirely (the live API's own `messages` field says
    so explicitly) - recipient_name (an approximate text match) is the
    only recipient-scoping option this specific tool has.

    Unlike the other two spending tools, award_type alone (which every
    call here has - it defaults to "contracts") counts as real scope
    (award_type_counts_as_scope=True), so this never raises for missing
    scope the way get_spending_by_category/get_spending_over_time can -
    see _build_filters' docstring and #125.
    """
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        time_period_type,
        start_year,
        end_year,
        award_type=award_type,
        recipient_name=recipient_name,
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
        tas_code=tas_code,
        federal_account=federal_account,
        def_codes=def_codes,
        award_type_counts_as_scope=True,
    )
    amount_field = _amount_field_for_award_type(award_type)
    sort_field = _sort_field_for_award_type(award_type, sort_by)
    # "Start Date" (period of performance) is fetched unconditionally so search_awards
    # can flag when a result's shown amount is a multi-year lifetime total that predates
    # the requested range, not spending scoped to it - see the overlap-vs-action-date
    # caveat in search_awards's docstring and issue #118.
    #
    # DISASTER_BREAKOUT_FIELDS are also fetched unconditionally - Base fields present
    # on every award regardless of whether def_codes is filtered on.
    fields = SEARCH_AWARDS_FIELDS_BASE + DISASTER_BREAKOUT_FIELDS + [amount_field, "Start Date"]
    if sort_field != amount_field:
        fields = fields + [sort_field]
    return client.search_awards(filters, fields=fields, limit=limit, sort=sort_field, order="desc")


@beta_tool
def search_awards(
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
) -> str:
    """Search for individual award records (specific contracts, grants, or loans) for a fiscal year range, scoped by an awarding agency and/or a recipient. Use this for "show me awards/contracts/grants from X" or "who received money from X" questions — as opposed to an aggregate breakdown or trend, which get_spending_by_category / get_spending_over_time answer instead. Results are ranked largest-first by sort_by (default "amount") — use this directly for "biggest"/"top N" questions, including "top N by outlay/subsidy cost" or "most recently modified" with sort_by set accordingly.

    No filter beyond the fiscal-year range and award_type is required - unlike
    get_spending_by_category/get_spending_over_time, which need a real scoping
    filter (agency, recipient, location, etc.) or they refuse to run at all, this
    tool just returns a ranked/paginated list, which works fine with an
    award-type + fiscal-year filter alone. Add agency_name/recipient_name/a
    location/etc. to narrow further, but
    don't invent a keywords value (e.g. restating award_type itself) just to
    satisfy a scope requirement this tool doesn't actually have.

    IMPORTANT about fiscal-year scoping: by default (date_type omitted), an
    award appears here if it had ANY transaction/modification in the queried
    fiscal year - not "this dollar amount was specifically obligated in this
    year." Award Amount/Loan Value is each award's current TOTAL value, not a
    per-year figure. A multi-year award active in more than one fiscal year
    will appear in results for EACH of those years showing the SAME total -
    do not sum or compare these across multiple fiscal-year calls to this
    tool as if they were period-scoped and additive; use
    get_spending_over_time for genuinely period-scoped, non-duplicative
    totals instead. If the question is really "what NEW contracts did X
    award in FY2024" rather than "what was X active on," set
    date_type="new_awards_only" instead of leaving this default.

    Each result also includes an internal_id - a separate, longer identifier
    from the plain Award ID (PIID/FAIN) shown alongside it. It isn't
    meaningful to a user, so don't quote it in your answer - pass it to
    get_award_details for full details on that specific award.

    A zero-results response names other award type categories worth trying -
    this tool only ever searches ONE award type per call, so an empty result
    means nothing had that specific type, never that the recipient/program has
    no award records at all. Follow that suggestion rather than concluding
    from one or two empty calls that a program isn't tracked as individual
    awards (e.g. a benefit/entitlement program is very likely
    direct_payment_specified/direct_payment_unrestricted, not grants or
    contracts).

    Args:
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 (Oct 2020-Sep 2021) or CY2021 (Jan-Dec 2021). Data is only
            available from FY2008 (or CY2007) onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        award_type: The broad buckets are contracts, grants, loans (default contracts) - use one
            of these for a general "show me X's contracts/grants" question. For a question asking
            about a SPECIFIC sub-type rather than the broad category, use the specific value
            instead of guessing which broad bucket it falls under: bpa_call, purchase_order,
            delivery_order, definitive_contract (contract sub-types); idv (Indefinite Delivery
            Vehicle - a GWAC, BPA, or other contract vehicle that other awards get issued under,
            NOT itself under "contracts" - its codes are disjoint from A/B/C/D, so searching for
            an IDV's own PIID under award_type="contracts" returns zero results); direct_loan,
            guaranteed_loan (loan sub-types); block_grant, formula_grant, project_grant,
            cooperative_agreement (grant sub-types - e.g. "cooperative agreement" is
            cooperative_agreement, NOT contracts); insurance, other_financial_assistance,
            direct_payment_specified, direct_payment_unrestricted (other assistance types).
            Case/spacing/hyphens don't matter (e.g. "Cooperative Agreement" also works).
        limit: Max number of results to return (default 5).
        sort_by: One of: amount (default - Award Amount, or Loan Value for loan award
            types), outlays (Total Outlays - the amount actually paid out so far, distinct
            from the obligated amount "amount" sorts by; NOT valid for loan award types,
            which have no such field - use subsidy_cost for loans instead), subsidy_cost
            (the government's actual budgetary cost of a loan, distinct from Loan Value's
            face value; ONLY valid for loan award types), recency (Last Modified Date - use
            for "most recently modified/updated award to X" questions). Whichever field is
            sorted on is also shown in each result line, not just used silently for ordering.
        agency_name: Optional. The awarding agency's name, e.g. "National Science
            Foundation". Omit for a cross-agency question about one recipient - but then
            recipient_name must be set instead.
        recipient_name: Optional. Restrict to awards whose recipient name contains this
            text, e.g. "Leidos". This is an approximate text match, the only recipient
            filter this tool supports (unlike get_spending_by_category/get_spending_over_time,
            this tool does not accept a recipient_id - confirmed live it's silently
            ignored here) - confirmed live it can both miss real subsidiaries whose legal
            name differs from the parent company's, and sweep in unrelated similarly-named
            entities (e.g. an unrelated joint venture).
        min_amount: Optional. Restrict to awards worth at least this dollar amount.
        max_amount: Optional. Restrict to awards worth at most this dollar amount.
        performed_in_state: Optional. Restrict to awards for work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to awards whose recipient is
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different results.
        performed_in_county: Optional. A specific county where work was performed - a 3-digit
            FIPS code (e.g. "025" for Yavapai County, AZ), not a name. Requires
            performed_in_state also be set. Use resolve_county_fips to find the code from a
            county name - do not guess or construct one.
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
        date_type: Optional. See the IMPORTANT note above - one of action_date (default),
            date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511". Must be
            the real code - if you only have a description, browse via get_spending_by_category
            (category="naics") instead of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
        award_id: Optional. Restrict to a single known award by its plain Award ID (PIID/FAIN/URI,
            e.g. "1605SS17F00018") - a fuzzy text match, not an exact-id lookup. This is the same
            "Award ID" shown in this tool's own results, NOT the longer internal_id shown alongside
            it (that one is for get_award_details, not this filter). If the PIID is for a contract
            vehicle (IDV) rather than a plain contract, pass award_type="idv" too - the default
            award_type="contracts" won't find it, since IDV codes are disjoint from A/B/C/D.
        recipient_type: Optional. Restrict to recipients tagged with this business/recipient
            type, e.g. "small_business", "woman_owned_business", "nonprofit", "higher_education".
        description: Optional. Restrict to awards whose own description text matches this
            phrase, e.g. "vaccine research". Distinct from keywords - keywords also matches
            recipient name, PIID/FAIN/URI, and NAICS/PSC description text, so a keywords hit
            doesn't imply a description hit or vice versa.
        tas_code: Optional. Restrict to awards funded by this exact Treasury Account Symbol,
            e.g. "020-2020/2021-1521". Must be the real code - get it from a
            get_award_funding_breakdown call, not guessed.
        federal_account: Optional. Restrict to awards funded by this exact federal account
            (the AID-MAIN pair one level up from a full TAS), e.g. "028-8704" - the
            federal_account value shown on a get_award_funding_breakdown row. Different from
            tas_code: a federal account groups multiple TAS together.
        def_codes: Optional. Restrict to awards tagged with these Disaster Emergency Fund
            Codes (DEFC), e.g. ["L"] for a single code. Use the group alias "covid" (or
            "covid_19") or "infrastructure" (or "iija") instead of listing every individual
            code in that group - e.g. def_codes=["covid"] covers all 7 COVID-relief codes.
            Sufficient scope on its own. Every result also reports its own def_codes plus
            COVID-19/Infrastructure Obligations and Outlays when non-zero, regardless of
            whether this filter is set.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    # Wrapper-level, not inside search_awards_raw - see the identical
    # comment on get_spending_by_category's clamp for why.
    limit = _clamp_limit(limit)
    scope = _scope_label(
        agency_name, recipient_name, None,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
        award_id=award_id, description=description,
        tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
    )
    try:
        results = search_awards_raw(
            agency_name,
            time_period_type,
            start_year,
            end_year,
            award_type,
            limit,
            sort_by=sort_by,
            recipient_name=recipient_name,
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
            tas_code=tas_code,
            federal_account=federal_account,
            def_codes=def_codes,
        )
    except USASpendingAPIError as e:
        logger.warning("search_awards failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_year": start_year,
            "end_year": end_year,
            "time_period_type": time_period_type,
            "award_type": award_type,
            "sort_by": sort_by,
        },
        agency_name=agency_name,
        recipient_name=recipient_name,
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
        tas_code=tas_code,
        federal_account=federal_account,
        def_codes=def_codes,
    )
    _record_tool_call("search_awards", results, context)

    if not results.results:
        others = _other_award_type_categories_to_try(award_type)
        return (
            f"No {award_type} awards found for {scope} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}. "
            f"This does NOT mean no award records exist for this recipient/program - only that none are "
            f"of type '{award_type}'. Before concluding there are no individual award records, try one or "
            f"more of the other award type categories: {others}."
        )

    amount_field = _amount_field_for_award_type(award_type)
    sort_field = _sort_field_for_award_type(award_type, sort_by)
    # Award-summary results OVERLAP the requested range rather than being scoped to it
    # (confirmed live, see #118 and fedspendingtransparency/usaspending-api#1707): an
    # award that started before start_year still shows its full lifetime amount,
    # not spending specific to this range. Flag any result whose period of performance
    # started before the range so that isn't presented as if it were period-scoped.
    range_start_date, _ = year_range_to_date_range(time_period_type, start_year, end_year)
    any_predates_range = False
    lines = []
    for r in results.results:
        result_award_id = r.get("Award ID", "unknown")
        internal_id = r.get("generated_internal_id", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        amount = r.get(amount_field)
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        sort_str = ""
        if sort_field != amount_field:
            sort_value = r.get(sort_field)
            sort_value_str = (
                f"${sort_value:,.2f}" if isinstance(sort_value, (int, float)) else str(sort_value)
            )
            sort_str = f", {sort_field}: {sort_value_str}"
        predates_range = isinstance((start_date := r.get("Start Date")), str) and start_date < range_start_date
        flag_str = ""
        if predates_range:
            any_predates_range = True
            flag_str = f" [PERIOD OF PERFORMANCE STARTED {start_date}, BEFORE {year_label(time_period_type, start_year)} - amount shown is this award's lifetime total, not spending scoped to this range]"
        disaster_str = ""
        result_def_codes = r.get("def_codes")
        # Only shown when non-empty - most awards carry no disaster tag at all.
        if result_def_codes:
            covid_obligations = r.get("COVID-19 Obligations") or 0
            covid_outlays = r.get("COVID-19 Outlays") or 0
            infra_obligations = r.get("Infrastructure Obligations") or 0
            infra_outlays = r.get("Infrastructure Outlays") or 0
            disaster_str = f" [DEFC: {', '.join(result_def_codes)}"
            if covid_obligations or covid_outlays:
                disaster_str += f"; COVID-19 Obligations: ${covid_obligations:,.2f}, COVID-19 Outlays: ${covid_outlays:,.2f}"
            if infra_obligations or infra_outlays:
                disaster_str += f"; Infrastructure Obligations: ${infra_obligations:,.2f}, Infrastructure Outlays: ${infra_outlays:,.2f}"
            disaster_str += "]"
        lines.append(
            f"{result_award_id} — {recipient}: {amount_str}{sort_str} [internal_id: {internal_id}]{flag_str}{disaster_str}"
        )
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results)) + _format_api_messages(results.messages)
    if any_predates_range:
        note += (
            "\n\nCAVEAT: one or more awards above started before the requested fiscal year range. "
            "This tool returns an award if it had ANY activity during the requested range, but always "
            "shows that award's full lifetime total/outlays - not the amount specific to this range. "
            "State this distinction explicitly if reporting these figures as this period's spending; "
            "use get_spending_over_time instead for a genuinely period-scoped, non-duplicative total."
        )
    return _wrap_untrusted("\n".join(lines) + note)


@traceable(run_type="tool", name="search_subawards_raw")
def search_subawards_raw(
    agency_name: str | None,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    subrecipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    subrecipient_in_state: str | None = None,
    performed_in_county: str | None = None,
    subrecipient_in_county: str | None = None,
    performed_in_city: str | None = None,
    subrecipient_in_city: str | None = None,
    performed_in_zip: str | None = None,
    subrecipient_in_zip: str | None = None,
    performed_in_district: str | None = None,
    subrecipient_in_district: str | None = None,
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
) -> SearchAwardsResponse:
    """Call the same live endpoint search_awards_raw uses, with
    spending_level="subawards" - confirmed live this returns individual
    subaward records (not prime awards) from the same
    search/spending_by_award/ endpoint, not a separate one.

    subrecipient_name/subrecipient_in_state/subrecipient_in_county/
    subrecipient_in_city/subrecipient_in_zip/subrecipient_in_district all
    filter the SUB-recipient - confirmed live ("Thermo Electron" as
    recipient_search_text correctly matched subawards TO Thermo Electron,
    not FROM it). performed_in_* still means the subaward's own place of
    performance, unchanged.

    recipient_type is NOT reversed the way subrecipient_name is - live-verified
    2026-09-12 that recipient_type_names still matches the PRIME recipient's
    business categories in subawards mode (e.g. recipient_type="veteran_
    owned_business" returned subawards whose Prime Recipient Name was a
    veteran-owned firm, sub-recipient unconstrained). description IS
    subaward-scoped, matching SUBAWARD_FIELDS' "Sub-Award Description" -
    confirmed from the live filter's own field mapping (subaward_description,
    not the prime award's description).

    recipient_id is deliberately not a parameter here (same as
    search_awards_raw) - confirmed live it's silently ignored for
    subawards, matching the API's own `messages` field.
    """
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        time_period_type,
        start_year,
        end_year,
        award_type=award_type,
        recipient_name=subrecipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=subrecipient_in_state,
        performed_in_county=performed_in_county,
        recipient_in_county=subrecipient_in_county,
        performed_in_city=performed_in_city,
        recipient_in_city=subrecipient_in_city,
        performed_in_zip=performed_in_zip,
        recipient_in_zip=subrecipient_in_zip,
        performed_in_district=performed_in_district,
        recipient_in_district=subrecipient_in_district,
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
    return client.search_awards(
        filters, fields=SUBAWARD_FIELDS, limit=limit, sort="Sub-Award Amount", order="desc",
        spending_level="subawards",
    )


@beta_tool
def search_subawards(
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    agency_name: str | None = None,
    subrecipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    subrecipient_in_state: str | None = None,
    performed_in_county: str | None = None,
    subrecipient_in_county: str | None = None,
    performed_in_city: str | None = None,
    subrecipient_in_city: str | None = None,
    performed_in_zip: str | None = None,
    subrecipient_in_zip: str | None = None,
    performed_in_district: str | None = None,
    subrecipient_in_district: str | None = None,
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
    """Search for individual SUBAWARD records - money a prime awardee passed on to a sub-recipient to do part of the work. Use this for "who did X subcontract to" or "what subawards has agency Y's spending generated" questions about subawards in general, scoped by an awarding agency and/or a sub-recipient. For the subawards under one SPECIFIC prime award already found via search_awards, use get_award_subawards instead - this tool searches across many awards, not one award's own list.

    Each result's internal_id is the PRIME award's internal_id (not a subaward-specific id) - pass it to get_award_details or get_award_subawards for the prime award's own full detail or its complete subaward list.

    At least one of agency_name or subrecipient_name must be given - a query scoped
    by neither would mean all federal subawards, ever, which this tool refuses
    rather than silently running.

    Args:
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 (Oct 2020-Sep 2021) or CY2021 (Jan-Dec 2021). Data is only
            available from FY2008 (or CY2007) onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        award_type: The broad buckets are contracts, grants, loans (default contracts) - same
            vocabulary as search_awards's award_type, applied to the underlying prime award's type.
        limit: Max number of results to return (default 5).
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
            Omit for a cross-agency question about one sub-recipient - but then subrecipient_name
            must be set instead.
        subrecipient_name: Optional. Restrict to subawards received by a sub-recipient whose name
            contains this text, e.g. "Thermo Electron" - an approximate text match. This filters
            the SUB-recipient, the entity that received the subaward, not the prime awardee.
        min_amount: Optional. Restrict to subawards worth at least this dollar amount.
        max_amount: Optional. Restrict to subawards worth at most this dollar amount.
        performed_in_state: Optional. Restrict to subawards for work performed in this US state.
        subrecipient_in_state: Optional. Restrict to subawards whose SUB-recipient is
            headquartered/located in this US state - not the prime.
        performed_in_county: Optional. A specific county where work was performed - a 3-digit
            FIPS code (e.g. "025" for Yavapai County, AZ), not a name. Requires
            performed_in_state also be set. Use resolve_county_fips to find the code from a
            county name - do not guess or construct one.
        subrecipient_in_county: Optional. Same as performed_in_county, but for the SUB-recipient's
            location. Requires subrecipient_in_state also be set.
        performed_in_city: Optional. Restrict to work performed in this city, e.g. "Livermore".
        subrecipient_in_city: Optional. Same as performed_in_city, but for the SUB-recipient's location.
        performed_in_zip: Optional. Restrict to work performed in this 5-digit zip code.
        subrecipient_in_zip: Optional. Restrict to a SUB-recipient located in this 5-digit zip code.
        performed_in_district: Optional. A specific congressional district where work was
            performed - a 2-digit number (e.g. "01"), paired with performed_in_state.
        subrecipient_in_district: Optional. Same as performed_in_district, but for the SUB-recipient's
            location, paired with subrecipient_in_state.
        keywords: Optional. Free-text search over subaward descriptions, e.g. "climate research".
            Do NOT restate the award_type/category itself here (e.g. "grant", "contracts",
            "cooperative agreement") - award_type already scopes that precisely, and doing so
            on top silently narrows results to only the subawards whose description text
            happens to contain that literal word. Rejected with an error if given alone.
        date_type: Optional. Which date the fiscal-year range is matched against - one of
            action_date (default), date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the SUB-recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511".
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number, format NN.NNN.
        award_id: Optional. Restrict to subawards under a single known award by its plain Award ID
            (PIID/FAIN/URI, e.g. "1605SS17F00018") - a fuzzy text match. This is the PRIME award's
            ID, not a subaward-specific id.
        recipient_type: Optional. Restrict to subawards whose PRIME recipient is tagged with this
            business/recipient type, e.g. "small_business", "veteran_owned_business" - unlike
            subrecipient_name above, this is NOT reversed to the sub-recipient (live-verified). Not
            sufficient scope on its own (like award_type).
        description: Optional. Restrict to subawards whose own description text matches this
            phrase, e.g. "climate research" - this IS the sub-award's own description, unlike
            recipient_type above. Distinct from keywords, which also matches other text fields.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    limit = _clamp_limit(limit)
    scope = _scope_label(
        agency_name, subrecipient_name, None,
        performed_in_state=performed_in_state, recipient_in_state=subrecipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=subrecipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=subrecipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=subrecipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=subrecipient_in_district,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
        award_id=award_id, description=description,
    )
    try:
        results = search_subawards_raw(
            agency_name,
            time_period_type,
            start_year,
            end_year,
            award_type,
            limit,
            subrecipient_name=subrecipient_name,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            subrecipient_in_state=subrecipient_in_state,
            performed_in_county=performed_in_county,
            subrecipient_in_county=subrecipient_in_county,
            performed_in_city=performed_in_city,
            subrecipient_in_city=subrecipient_in_city,
            performed_in_zip=performed_in_zip,
            subrecipient_in_zip=subrecipient_in_zip,
            performed_in_district=performed_in_district,
            subrecipient_in_district=subrecipient_in_district,
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
        logger.warning("search_subawards failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_year": start_year, "end_year": end_year,
            "time_period_type": time_period_type, "award_type": award_type,
        },
        agency_name=agency_name,
        recipient_name=subrecipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=subrecipient_in_state,
        performed_in_county=performed_in_county,
        recipient_in_county=subrecipient_in_county,
        performed_in_city=performed_in_city,
        recipient_in_city=subrecipient_in_city,
        performed_in_zip=performed_in_zip,
        recipient_in_zip=subrecipient_in_zip,
        performed_in_district=performed_in_district,
        recipient_in_district=subrecipient_in_district,
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
    _record_tool_call("search_subawards", results, context)

    if not results.results:
        others = _other_award_type_categories_to_try(award_type)
        return (
            f"No {award_type} subawards found for {scope} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}. "
            f"This does NOT mean no subaward records exist - only that none are under a '{award_type}'-type "
            f"prime award. Before concluding there are no subaward records, try one or more of the other "
            f"award type categories: {others}."
        )

    lines = []
    for r in results.results:
        sub_id = r.get("Sub-Award ID", "unknown")
        sub_recipient = r.get("Sub-Awardee Name", "unknown")
        amount = r.get("Sub-Award Amount")
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        prime_award_id = r.get("Prime Award ID", "unknown")
        prime_recipient = r.get("Prime Recipient Name", "unknown")
        prime_internal_id = r.get("prime_award_generated_internal_id", "unknown")
        lines.append(
            f"{sub_id} — {sub_recipient}: {amount_str} (subaward under prime {prime_award_id} "
            f"from {prime_recipient}) [internal_id: {prime_internal_id}]"
        )
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results)) + _format_api_messages(results.messages)
    return _wrap_untrusted("\n".join(lines) + note)


@traceable(run_type="tool", name="search_transactions_raw")
def search_transactions_raw(
    agency_name: str | None,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    sort_by: TransactionSortBy = "amount",
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
) -> SearchAwardsResponse:
    """Call the API once, return the structured response - same filter
    resolution (_build_filters, award_type_counts_as_scope=True) as
    search_awards_raw, but hits search/spending_by_transaction/ instead of
    search/spending_by_award/, so each result is one transaction/
    modification rather than one award's cumulative lifetime total (issue
    #23). See client.search_transactions's docstring for the endpoint
    contract."""
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        time_period_type,
        start_year,
        end_year,
        award_type=award_type,
        recipient_name=recipient_name,
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
        tas_code=tas_code,
        federal_account=federal_account,
        def_codes=def_codes,
        award_type_counts_as_scope=True,
    )
    amount_field = _transaction_amount_field_for_award_type(award_type)
    sort_field = amount_field if sort_by == "amount" else "Action Date"
    fields = TRANSACTION_FIELDS_BASE + [amount_field]
    return client.search_transactions(filters, fields=fields, limit=limit, sort=sort_field, order="desc")


@beta_tool
def search_transactions(
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    sort_by: TransactionSortBy = "amount",
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
) -> str:
    """Search for individual transaction/modification records - one row per transaction (not per award), matching USASpending.gov's own "Keyword Search" results page. Use this for "show me each modification/transaction for X" or "what individual transactions match keyword Y" questions where the user wants the mod-by-mod history across many awards, as opposed to search_awards, which returns one row per award with that award's cumulative lifetime total - a single multi-year award with 10 modifications appears as ONE row there but up to 10 separate rows here, each with its own action date and amount.

    Same filter set as search_awards (agency/recipient/location/amount/etc.), and the same fiscal-year overlap caveat applies: an award's transactions/mods are matched by date_type (action_date by default), not by whether the award itself started in the requested range.

    Args:
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 (Oct 2020-Sep 2021) or CY2021 (Jan-Dec 2021). Data is only
            available from FY2008 (or CY2007) onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        award_type: The broad buckets are contracts, grants, loans (default contracts) - same
            vocabulary as search_awards's award_type; see search_awards's own docstring for
            the full list of specific sub-types (bpa_call, idv, cooperative_agreement, etc.).
        limit: Max number of transaction rows to return (default 5) - a single award can
            contribute more than one row if it has multiple mods in range.
        sort_by: "amount" (default - Transaction Amount, or Loan Value for loan award types)
            or "recency" (Action Date, most recent first).
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
        recipient_name: Optional. Restrict to transactions whose recipient name contains this
            text, e.g. "Leidos" - an approximate text match, same as search_awards.
        min_amount: Optional. Restrict to transactions worth at least this dollar amount -
            a per-transaction amount, not an award's lifetime total.
        max_amount: Optional. Restrict to transactions worth at most this dollar amount.
        performed_in_state: Optional. Restrict to work performed in this US state.
        recipient_in_state: Optional. Restrict to a recipient headquartered/located in this
            US state - different from performed_in_state.
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
        keywords: Optional. Free-text search over transaction descriptions, e.g. "ventilators" -
            this is the parameter that matches USASpending.gov's own Keyword Search box.
        date_type: Optional. Which date the fiscal-year range is matched against - one of
            action_date (default), date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511".
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
        award_id: Optional. Restrict to transactions under a single known award by its plain
            Award ID (PIID/FAIN/URI) - a fuzzy text match. For the FULL mod-by-mod history of
            one SPECIFIC award already resolved to its internal_id, get_award_transaction_history
            is more direct than filtering this tool by award_id.
        recipient_type: Optional. Restrict to recipients tagged with this business/recipient
            type, e.g. "small_business", "woman_owned_business", "nonprofit", "higher_education".
        description: Optional. Restrict to transactions whose own description text matches this
            phrase - distinct from keywords, which also matches recipient name/PIID/FAIN/NAICS/PSC.
        tas_code: Optional. Restrict to transactions funded by this exact Treasury Account
            Symbol, e.g. "020-2020/2021-1521".
        federal_account: Optional. Restrict to transactions funded by this exact federal
            account (the AID-MAIN pair one level up from a full TAS), e.g. "028-8704".
        def_codes: Optional. Restrict to transactions tagged with these Disaster Emergency
            Fund Codes (DEFC), e.g. ["L"]. Use the group alias "covid" or "infrastructure"
            instead of listing every individual code in that group.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    limit = _clamp_limit(limit)
    scope = _scope_label(
        agency_name, recipient_name, None,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        performed_in_county=performed_in_county, recipient_in_county=recipient_in_county,
        performed_in_city=performed_in_city, recipient_in_city=recipient_in_city,
        performed_in_zip=performed_in_zip, recipient_in_zip=recipient_in_zip,
        performed_in_district=performed_in_district, recipient_in_district=recipient_in_district,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
        award_id=award_id, description=description,
        tas_code=tas_code, federal_account=federal_account, def_codes=def_codes,
    )
    try:
        results = search_transactions_raw(
            agency_name,
            time_period_type,
            start_year,
            end_year,
            award_type,
            limit,
            sort_by=sort_by,
            recipient_name=recipient_name,
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
            tas_code=tas_code,
            federal_account=federal_account,
            def_codes=def_codes,
        )
    except USASpendingAPIError as e:
        logger.warning("search_transactions failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_year": start_year,
            "end_year": end_year,
            "time_period_type": time_period_type,
            "award_type": award_type,
            "sort_by": sort_by,
        },
        agency_name=agency_name,
        recipient_name=recipient_name,
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
        tas_code=tas_code,
        federal_account=federal_account,
        def_codes=def_codes,
    )
    _record_tool_call("search_transactions", results, context)

    if not results.results:
        others = _other_award_type_categories_to_try(award_type)
        return (
            f"No {award_type} transactions found for {scope} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}. "
            f"This does NOT mean no transaction records exist for this recipient/program - only that none are "
            f"of type '{award_type}'. Before concluding there are no transaction records, try one or "
            f"more of the other award type categories: {others}."
        )

    amount_field = _transaction_amount_field_for_award_type(award_type)
    lines = []
    for r in results.results:
        award_id_val = r.get("Award ID", "unknown")
        mod = r.get("Mod", "?")
        internal_id = r.get("generated_internal_id", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        action_date = r.get("Action Date", "unknown")
        amount = r.get(amount_field)
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        lines.append(
            f"{award_id_val} Mod {mod} — {recipient}: {amount_str} on {action_date} [internal_id: {internal_id}]"
        )
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results)) + _format_api_messages(results.messages)
    return _wrap_untrusted("\n".join(lines) + note)

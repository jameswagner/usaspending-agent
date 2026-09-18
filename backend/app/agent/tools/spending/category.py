"""get_spending_by_category - spending broken down by a category (agency,
NAICS/PSC code, recipient, geography, ...), ranked by amount. One of the
five _build_filters-based tools split out of the old flat spending.py (#204).
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    SpendingByCategoryResponse,
    USASpendingAPIError,
)

from ...recipient_types import RecipientType
from ...response_shaping import year_label
from ...singletons import _get_usaspending_client
from ...tool_filters import (
    AwardType,
    DateType,
    Scope,
    _build_filters,
    _clamp_limit,
    _record_optional_filter_context,
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

VALID_CATEGORIES = {
    "awarding_agency", "awarding_subagency", "cfda", "country", "county",
    "defc", "district", "federal_account", "funding_agency",
    "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
    "state_territory",
}

# Static Literal mirror of VALID_CATEGORIES, used as get_spending_by_category's
# actual parameter type so beta_tool's schema generation emits a real
# JSON-schema enum - same reasoning as AwardType in tool_filters.py.
# TestLiteralTypesMatchVocabulary (tests/test_agent.py) asserts this can't
# silently drift from VALID_CATEGORIES.
Category = Literal[
    "awarding_agency", "awarding_subagency", "cfda", "country", "county",
    "defc", "district", "federal_account", "funding_agency",
    "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
    "state_territory",
]

SpendingLevel = Literal["transactions", "awards", "subawards", "award_financial"]


def _normalize_category(category: str) -> str:
    """Same normalize-then-validate pattern as _normalize_award_type/
    _normalize_state: raises a clean, actionable USASpendingAPIError
    instead of letting an unrecognized category reach the live API and
    come back as a bare 404 with no guidance on what would have worked."""
    normalized = category.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in VALID_CATEGORIES:
        raise USASpendingAPIError(
            f"Unknown category '{category}'. Must be one of: {', '.join(sorted(VALID_CATEGORIES))}"
        )
    return normalized


@traceable(run_type="tool", name="get_spending_by_category_raw")
def get_spending_by_category_raw(
    category: Category,
    agency_name: str | None,
    time_period_type: Literal["fiscal", "calendar"],
    start_year: int,
    end_year: int,
    limit: int = 5,
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
    spending_level: SpendingLevel = "transactions",
) -> SpendingByCategoryResponse:
    """Call the API once, return the structured response. Raises
    USASpendingAPIError on failure — the @beta_tool wrapper decides how to
    present that to the model; this function stays presentation-free so the
    structured result is also available for chart-building later.

    category is validated against VALID_CATEGORIES before ever reaching the
    live API - same "code owns the exact vocabulary" pattern as award_type.

    agency_name is optional - see _build_filters' docstring
    for why (a cross-agency, recipient-only question has no answer at all
    if an agency must always be named), and for why recipient_id (an
    exact, precise filter - unlike recipient_name's text match) is only
    ever wired through here and on get_spending_over_time, never
    search_awards.

    Filter resolution (agency, award_type, recipient, amount, location,
    keywords, date_type, scope, naics/psc/cfda code) is delegated to
    _build_filters - see its docstring for the "no behavior change when
    the new params are omitted" guarantee and each failure mode.
    """
    category = _normalize_category(category)
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
    return client.spending_by_category(category, filters, limit=limit, spending_level=spending_level)


@beta_tool
def get_spending_by_category(
    category: Category,
    *,
    time_period_type: Literal["fiscal", "calendar"] = "fiscal",
    start_year: int,
    end_year: int,
    limit: int = 5,
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
    spending_level: SpendingLevel = "transactions",
) -> str:
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for a fiscal year range, scoped by a real scoping filter, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions. Returns only the top `limit` categories, not a grand total - for "what is the total/how much funding" questions, use get_spending_over_time instead (grouped by fiscal_year), never this tool's top-N rows.

    At least one real scoping filter must be given - agency_name, recipient_name,
    recipient_id, a location filter (performed_in_*/recipient_in_*), naics_code, psc_code,
    cfda_program, keywords, award_id, description, or recipient_type. A query scoped by
    none of them would mean all federal spending, ever, which this tool refuses rather than
    silently running.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient, recipient_duns, state_territory. Enforced in code - any other value (including ones the API's own docs list, like object_class or tas, which 404 in practice) fails cleanly with this exact list rather than reaching the live API. recipient and recipient_duns return the same results for every case tested - either works for "top recipients" questions.
        time_period_type: "fiscal" (default) for federal fiscal years (Oct-Sep, named by the
            year they end in) or "calendar" for plain Jan-Dec calendar years. Use "calendar"
            when the user explicitly says "calendar year"/"CY2023" or asks about a plain
            Jan-Dec window; default to "fiscal" otherwise.
        start_year: First year to include (fiscal or calendar per time_period_type above),
            e.g. 2021 for FY2021 (Oct 2020-Sep 2021) or CY2021 (Jan-Dec 2021). Data is only
            available from FY2008 (or CY2007) onward.
        end_year: Last year to include, e.g. 2024 for FY2024 or CY2024.
        limit: Max number of results to return (default 5).
        agency_name: Optional. The awarding agency's name, e.g. "National Science Foundation".
            Omit for a cross-agency question about one recipient (e.g. "which agencies has X
            received money from") - but then recipient_name or recipient_id must be set instead.
        award_type: Optional. Restrict to one award type or bucket - contracts, grants,
            loans, or a specific sub-type like cooperative_agreement (same values and
            case/spacing-insensitive matching as search_awards's award_type). Omit to
            include all award types.
        recipient_name: Optional. Restrict to spending from awards whose recipient name
            contains this text, e.g. "Leidos". This is an approximate text match - confirmed
            live that it can both miss real subsidiaries whose legal name differs from the
            parent company's, and sweep in unrelated similarly-named entities (e.g. an
            unrelated joint venture). If you already have a specific recipient_id (e.g. from
            search_recipients), pass that instead for an exact, correct total.
        recipient_id: Optional. The exact recipient_id from a prior search_recipients or
            get_recipient_details call (e.g. "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P") - an
            exact identifier, not a text match. Confirmed live to reproduce a recipient's true
            total precisely. Prefer this over recipient_name whenever you have it. Do not
            guess or construct a recipient_id.
        min_amount: Optional. Restrict to spending of at least this dollar amount.
        max_amount: Optional. Restrict to spending of at most this dollar amount.
        performed_in_state: Optional. Restrict to spending on work performed in this US
            state (where the work happened), e.g. "Virginia" or "VA".
        recipient_in_state: Optional. Restrict to spending on recipients
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different totals.
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
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default: any transaction/modification in the window - note this means
            a multi-year award active in more than one fiscal year appears in results for EACH
            of those years), date_signed (the award's original signing date), last_modified_date,
            or new_awards_only (only awards that originated in this window - use this for "what
            NEW contracts did X award in FY2024" as opposed to "what was X spending on in FY2024").
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511" for
            Custom Computer Programming Services. Must be the real code (2-6 digits) - if you
            only have a description, use category="naics" to browse the actual breakdown instead
            of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030"
            for Information Technology Software. Same guidance as naics_code: use category="psc"
            to browse if you don't have the exact code.
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
        award_id: Optional. Restrict to a single known award (PIID/FAIN/URI), e.g.
            "1605SS17F00018" - a fuzzy text match, not an exact-id lookup. Useful when you
            already have a specific award's ID (e.g. from search_awards) and want spending
            scoped to just that one award.
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
            Sufficient scope on its own. For a spending-by-DEFC breakdown instead of filtering
            to a specific one, use category="defc" instead.
        spending_level: Optional. The level of spending detail to aggregate by (default
            "transactions"). Use "subawards" to answer "total subaward dollars to district X"
            questions - confirmed live 2026-09-16 this returns an aggregate subaward total per
            category value (e.g. IL-01 -> one dollar amount), not individual subaward records -
            for that, use search_subawards instead. "award_financial" is confirmed live to only
            work with category="defc" - any other category returns a clean API error ("Category
            '<category>' is not implemented when 'spending_level' is 'award_financial'"), so
            don't combine it with any other category. "awards" and "transactions" (the default)
            both work with every category; the difference between them is not yet live-verified
            here.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    # Clamped here, at the model-facing boundary, not inside
    # get_spending_by_category_raw - the _raw function is the trusted
    # internal API (dev_tools scripts, tests can legitimately want an
    # uncapped value); this wrapper is the untrusted boundary a model's
    # tool call actually crosses, which is where the real abuse surface is.
    limit = _clamp_limit(limit)
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
        response = get_spending_by_category_raw(
            category,
            agency_name,
            time_period_type,
            start_year,
            end_year,
            limit,
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
            spending_level=spending_level,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_category failed for %s/%s: %s", scope, category, e)
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    context = _record_optional_filter_context(
        {
            "category": category,
            "start_year": start_year,
            "end_year": end_year,
            "time_period_type": time_period_type,
            "spending_level": spending_level,
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
    _record_tool_call("get_spending_by_category", response, context)

    if not response.results:
        return (
            f"No {category} spending data found for {scope} between "
            f"{year_label(time_period_type, start_year)} and {year_label(time_period_type, end_year)}."
        )

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
    return _wrap_untrusted("\n".join(lines) + note)

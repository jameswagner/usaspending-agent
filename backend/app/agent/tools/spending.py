"""The _build_filters-based cluster: get_spending_by_category,
get_spending_over_time, search_awards, get_spending_by_geography. All four
funnel their filter params through the shared tool_filters._build_filters.
"""
from __future__ import annotations

import logging
from typing import Literal

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending_client import (
    GeographyTypeResult,
    SearchAwardsResponse,
    SpendingByCategoryResponse,
    SpendingByGeographyResponse,
    SpendingOverTimeResponse,
    USASpendingAPIError,
)

from ..response_shaping import _format_time_period
from ..singletons import _get_usaspending_client
from ..tool_filters import (
    SEARCH_AWARDS_FIELDS_BASE,
    AwardType,
    DateType,
    GeoLayer,
    GeoScope,
    Scope,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
    _record_optional_filter_context,
)
from ._shared import (
    _check_tool_call_budget,
    _format_api_messages,
    _record_tool_call,
    _scope_label,
    _truncation_note,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)

# This is the real, live API's set of working categories, not the API
# contract's documented one - the contract lists 18, but 4 of them
# (object_class, program_activity, recipient_parent_duns, tas) 404 in
# practice despite being documented. Same "code owns the exact
# vocabulary, not the model" pattern as AWARD_TYPE_GROUPS/
# US_STATE_ABBREVIATIONS.
#
# "recipient" isn't in the API contract's own top-level category enum
# but works live. Kept alongside recipient_duns (identical results for
# every case tested) since recipient additionally carries a uei field
# and models a "MULTIPLE RECIPIENTS" rollup row that recipient_duns's
# schema doesn't.
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
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
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
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    return client.spending_by_category(category, filters, limit=limit)


@beta_tool
def get_spending_by_category(
    category: Category,
    start_fiscal_year: int,
    end_fiscal_year: int,
    limit: int = 5,
    agency_name: str | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Get USASpending spending broken down by a category (e.g. industry, product/service code, sub-agency) for a fiscal year range, scoped by an awarding agency and/or a recipient, ranked by total amount descending. Use this for "how is X's spending broken down by Y" questions.

    At least one of agency_name, recipient_name, or recipient_id must be given - a query
    scoped by none of them would mean all federal spending, ever, which this tool refuses
    rather than silently running.

    Args:
        category: One of: awarding_agency, awarding_subagency, cfda, country, county, defc, district, federal_account, funding_agency, funding_subagency, naics, psc, recipient, recipient_duns, state_territory. Enforced in code - any other value (including ones the API's own docs list, like object_class or tas, which 404 in practice) fails cleanly with this exact list rather than reaching the live API. recipient and recipient_duns return the same results for every case tested - either works for "top recipients" questions.
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
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
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
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
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
    )
    try:
        response = get_spending_by_category_raw(
            category,
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            limit,
            award_type=award_type,
            recipient_name=recipient_name,
            recipient_id=recipient_id,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_category failed for %s/%s: %s", scope, category, e)
        return f"This query failed: {e}. Do not substitute a different category and present it as answering the original question — tell the user this specific breakdown isn't available."

    context = _record_optional_filter_context(
        {
            "category": category,
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
        },
        agency_name=agency_name,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    _record_tool_call("get_spending_by_category", response, context)

    if not response.results:
        return f"No {category} spending data found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [f"{r.name or r.code or 'unknown'}: ${r.amount:,.2f}" for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results)) + _format_api_messages(response.messages)
    return _wrap_untrusted("\n".join(lines) + note)


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
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SpendingOverTimeResponse:
    """Call the API once, return the structured response. Same filter
    resolution (via _build_filters) as get_spending_by_category_raw -
    including agency_name's optionality and recipient_id's exactness, see
    that function's docstring."""
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        recipient_id=recipient_id,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    return client.spending_over_time(filters, group=_normalize_group(group))


@beta_tool
def get_spending_over_time(
    start_fiscal_year: int,
    end_fiscal_year: int,
    group: Group = "fiscal_year",
    agency_name: str | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Get USASpending spending trends over time for a fiscal year range, scoped by an awarding agency and/or a recipient, grouped by period. Use this for "how has X's spending changed/trended over time" questions.

    At least one of agency_name, recipient_name, or recipient_id must be given - a query
    scoped by none of them would mean all federal spending, ever, which this tool refuses
    rather than silently running.

    Args:
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        group: One of: fiscal_year, calendar_year, quarter, month. Default fiscal_year.
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
            headquartered/located in this US state - different from performed_in_state:
            a company headquartered in one state can perform work in another, and these
            can give substantially different totals.
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default: any transaction/modification in the window - note this means
            a multi-year award active in more than one fiscal year contributes to EACH of those
            years' totals), date_signed (the award's original signing date), last_modified_date,
            or new_awards_only (only awards that originated in this window).
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code, e.g. "541511". Must be
            the real code - if you only have a description, use get_spending_by_category with
            category="naics" to browse the actual breakdown instead of guessing a code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code, e.g. "7030".
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number (grants
            only), format NN.NNN, e.g. "10.001".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    scope = _scope_label(
        agency_name, recipient_name, recipient_id,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
    )
    try:
        response = get_spending_over_time_raw(
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            group,
            award_type=award_type,
            recipient_name=recipient_name,
            recipient_id=recipient_id,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_over_time failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
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
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    _record_tool_call("get_spending_over_time", response, context)

    if not response.results:
        return f"No spending-over-time data found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    lines = [
        f"{_format_time_period(r.time_period)}: ${r.aggregated_amount:,.2f}"
        for r in response.results
    ]
    return _wrap_untrusted("\n".join(lines) + _format_api_messages(response.messages))


@traceable(run_type="tool", name="search_awards_raw")
def search_awards_raw(
    agency_name: str | None,
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SearchAwardsResponse:
    """Call the API once, return the structured response (results +
    page_metadata), sorted largest-amount-first (Award Amount, or Loan
    Value for loan award types - see _amount_field_for_award_type). The
    live API's own default order is essentially arbitrary - an unsorted
    "top 5" NSF FY2023 contracts query once returned awards from $7K to
    $7.2M while the true largest that year ($3.13B) never appeared.
    Unlike the filter params, which are optional and behavior-preserving
    when omitted, this sort is NOT optional - there's no meaningful
    default ordering to preserve.

    Filter resolution (agency, award_type, recipient, amount, location) is
    delegated to _build_filters, same as the other two spending tools. No
    recipient_id param here, unlike the other two: this endpoint silently
    ignores that filter entirely (the live API's own `messages` field says
    so explicitly) - recipient_name (an approximate text match) is the
    only recipient-scoping option this specific tool has.
    """
    client = _get_usaspending_client()
    filters = _build_filters(
        client,
        agency_name,
        start_fiscal_year,
        end_fiscal_year,
        award_type=award_type,
        recipient_name=recipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        recipient_in_state=recipient_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
    )
    amount_field = _amount_field_for_award_type(award_type)
    fields = SEARCH_AWARDS_FIELDS_BASE + [amount_field]
    return client.search_awards(filters, fields=fields, limit=limit, sort=amount_field, order="desc")


@beta_tool
def search_awards(
    start_fiscal_year: int,
    end_fiscal_year: int,
    award_type: AwardType = "contracts",
    limit: int = 5,
    agency_name: str | None = None,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> str:
    """Search for individual award records (specific contracts, grants, or loans) for a fiscal year range, scoped by an awarding agency and/or a recipient. Use this for "show me awards/contracts/grants from X" or "who received money from X" questions — as opposed to an aggregate breakdown or trend, which get_spending_by_category / get_spending_over_time answer instead. Results are ranked largest-amount-first by default — use this directly for "biggest"/"top N" questions.

    At least one of agency_name or recipient_name must be given - a query scoped
    by neither would mean all federal awards, ever, which this tool refuses rather
    than silently running.

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

    Args:
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021 (Oct 2020-Sep 2021). Data is only available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
        award_type: The broad buckets are contracts, grants, loans (default contracts) - use one
            of these for a general "show me X's contracts/grants" question. For a question asking
            about a SPECIFIC sub-type rather than the broad category, use the specific value
            instead of guessing which broad bucket it falls under: bpa_call, purchase_order,
            delivery_order, definitive_contract (contract sub-types); direct_loan, guaranteed_loan
            (loan sub-types); block_grant, formula_grant, project_grant, cooperative_agreement
            (grant sub-types - e.g. "cooperative agreement" is cooperative_agreement, NOT
            contracts); insurance, other_financial_assistance, direct_payment_specified,
            direct_payment_unrestricted (other assistance types). Case/spacing/hyphens don't
            matter (e.g. "Cooperative Agreement" also works).
        limit: Max number of results to return (default 5).
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
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
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
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    # Wrapper-level, not inside search_awards_raw - see the identical
    # comment on get_spending_by_category's clamp for why.
    limit = _clamp_limit(limit)
    scope = _scope_label(
        agency_name, recipient_name, None,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
    )
    try:
        results = search_awards_raw(
            agency_name,
            start_fiscal_year,
            end_fiscal_year,
            award_type,
            limit,
            recipient_name=recipient_name,
            min_amount=min_amount,
            max_amount=max_amount,
            performed_in_state=performed_in_state,
            recipient_in_state=recipient_in_state,
            keywords=keywords,
            date_type=date_type,
            place_of_performance_scope=place_of_performance_scope,
            recipient_scope=recipient_scope,
            naics_code=naics_code,
            psc_code=psc_code,
            cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("search_awards failed for %s: %s", scope, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "start_fiscal_year": start_fiscal_year,
            "end_fiscal_year": end_fiscal_year,
            "award_type": award_type,
        },
        agency_name=agency_name,
        recipient_name=recipient_name,
        min_amount=min_amount,
        max_amount=max_amount,
        performed_in_state=performed_in_state,
        keywords=keywords,
        date_type=date_type,
        place_of_performance_scope=place_of_performance_scope,
        recipient_scope=recipient_scope,
        naics_code=naics_code,
        psc_code=psc_code,
        cfda_program=cfda_program,
        recipient_in_state=recipient_in_state,
    )
    _record_tool_call("search_awards", results, context)

    if not results.results:
        return f"No {award_type} awards found for {scope} between FY{start_fiscal_year} and FY{end_fiscal_year}."

    amount_field = _amount_field_for_award_type(award_type)
    lines = []
    for r in results.results:
        award_id = r.get("Award ID", "unknown")
        internal_id = r.get("generated_internal_id", "unknown")
        recipient = r.get("Recipient Name", "unknown")
        amount = r.get(amount_field)
        amount_str = f"${amount:,.2f}" if isinstance(amount, (int, float)) else "unknown amount"
        lines.append(f"{award_id} — {recipient}: {amount_str} [internal_id: {internal_id}]")
    has_next = results.page_metadata.hasNext if results.page_metadata else False
    note = _truncation_note(has_next, len(results.results)) + _format_api_messages(results.messages)
    return _wrap_untrusted("\n".join(lines) + note)


# Self-imposed, not API-driven - spending_by_geography has no limit/page
# param and returns every matching region (e.g. ~3,100+ for an unfiltered
# US county breakdown). Caps what reaches the model per call.
MAX_GEOGRAPHY_RESULTS = 20


@traceable(run_type="tool", name="get_spending_by_geography_raw")
def get_spending_by_geography_raw(
    scope: GeoScope,
    geo_layer: GeoLayer,
    start_fiscal_year: int,
    end_fiscal_year: int,
    agency_name: str | None = None,
    geo_layer_filters: list[str] | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
) -> SpendingByGeographyResponse:
    client = _get_usaspending_client()
    filters = _build_filters(
        client, agency_name, start_fiscal_year, end_fiscal_year,
        award_type=award_type, recipient_name=recipient_name, recipient_id=recipient_id,
        min_amount=min_amount, max_amount=max_amount,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        keywords=keywords, date_type=date_type,
        place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
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
    start_fiscal_year: int,
    end_fiscal_year: int,
    agency_name: str | None = None,
    geo_layer_filters: list[str] | None = None,
    award_type: AwardType | None = None,
    recipient_name: str | None = None,
    recipient_id: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    keywords: str | None = None,
    date_type: DateType | None = None,
    place_of_performance_scope: Scope | None = None,
    recipient_scope: Scope | None = None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    cfda_program: str | None = None,
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
        start_fiscal_year: First fiscal year to include, e.g. 2021 for FY2021. Data is only
            available from FY2008 onward.
        end_fiscal_year: Last fiscal year to include, e.g. 2024 for FY2024.
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
        keywords: Optional. Free-text search over award descriptions, e.g. "climate research".
        date_type: Optional. Which award date the fiscal-year range is matched against - one of
            action_date (default), date_signed, last_modified_date, or new_awards_only.
        place_of_performance_scope: Optional. "domestic" or "foreign" - where the work was performed.
        recipient_scope: Optional. "domestic" or "foreign" - where the recipient is located.
        naics_code: Optional. Restrict to this exact NAICS industry code.
        psc_code: Optional. Restrict to this exact 4-character Product/Service Code.
        cfda_program: Optional. Restrict to this exact CFDA/Assistance Listing number, format NN.NNN.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    scope_label_str = _scope_label(
        agency_name, recipient_name, recipient_id,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program, keywords=keywords,
    )
    try:
        response = get_spending_by_geography_raw(
            scope, geo_layer, start_fiscal_year, end_fiscal_year,
            agency_name=agency_name, geo_layer_filters=geo_layer_filters,
            award_type=award_type, recipient_name=recipient_name, recipient_id=recipient_id,
            min_amount=min_amount, max_amount=max_amount,
            performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
            keywords=keywords, date_type=date_type,
            place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
            naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
        )
    except USASpendingAPIError as e:
        logger.warning("get_spending_by_geography failed for %s: %s", scope_label_str, e)
        return f"This query failed: {e}."

    context = _record_optional_filter_context(
        {
            "scope": scope, "geo_layer": geo_layer,
            "start_fiscal_year": start_fiscal_year, "end_fiscal_year": end_fiscal_year,
        },
        agency_name=agency_name, award_type=award_type, recipient_name=recipient_name,
        recipient_id=recipient_id, min_amount=min_amount, max_amount=max_amount,
        performed_in_state=performed_in_state, recipient_in_state=recipient_in_state,
        keywords=keywords, date_type=date_type,
        place_of_performance_scope=place_of_performance_scope, recipient_scope=recipient_scope,
        naics_code=naics_code, psc_code=psc_code, cfda_program=cfda_program,
    )
    _record_tool_call("get_spending_by_geography", response, context)

    if not response.results:
        no_results = f"No spending-by-geography data found for {scope_label_str} between FY{start_fiscal_year} and FY{end_fiscal_year}."
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

"""The shared filter layer the three spending tools (get_spending_by_category,
get_spending_over_time, search_awards) in tools.py build on: award-type and
US-state vocabulary normalization, and _build_filters, which turns the
common agency/date/award-type/recipient/amount/location parameters into a
real AdvancedFilters object.

Split out of tools.py (2026-09-06) once it grew large enough to be doing two
different jobs - this module is purely mechanical, endpoint-agnostic filter
construction; tools.py is the five actual @beta_tool definitions that use
it. Nothing here depends on anything in tools.py, so this can be imported
freely by future tools without pulling in the @beta_tool/@traceable
machinery those don't need.
"""
from __future__ import annotations

from backend.app.usaspending_client import (
    AdvancedFilters,
    AgencyFilter,
    AwardAmount,
    LocationObject,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

from .response_shaping import fiscal_year_to_date_range

# Verified against USASpending's own award_types.md contract (checked
# 2026-09-06), not guessed. The three broad buckets stay for general
# "show me X's contracts/grants/loans" questions, but a bare 3-way
# classification isn't enough: found live that asked for NSF's
# "cooperative agreements," the model picked award_type="contracts" -
# not just a broader bucket than asked for, but the flat-out wrong one,
# since a cooperative agreement isn't a contract at all. The fix isn't a
# better docstring hint (still trusting the model to classify correctly
# from a paragraph of prose) - it's exposing the real, specific sub-types
# as their own values, the same "let code do the exact lookup" pattern as
# find_agency_by_name, so the model only has to recognize a term close to
# what it already is, not correctly classify it into a bucket first.
#
# IDV-family codes (IDV_A through IDV_E - GWACs, BOAs, BPAs, etc.) are
# deliberately not included: those are a structurally different kind of
# award record (a vehicle other awards get issued under, not a
# transaction itself), and search_awards's field set/behavior for that
# category hasn't been verified - a real scope limitation, not an
# oversight, flagged here rather than silently extended to cover it.
AWARD_TYPE_GROUPS: dict[str, list[str]] = {
    "contracts": ["A", "B", "C", "D"],
    "grants": ["02", "03", "04", "05"],
    "loans": ["07", "08"],
    "bpa_call": ["A"],
    "purchase_order": ["B"],
    "delivery_order": ["C"],
    "definitive_contract": ["D"],
    "direct_loan": ["07"],
    "guaranteed_loan": ["08"],
    "block_grant": ["02"],
    "formula_grant": ["03"],
    "project_grant": ["04"],
    "cooperative_agreement": ["05"],
    "insurance": ["09"],
    "other_financial_assistance": ["11"],
    "direct_payment_specified": ["06"],
    "direct_payment_unrestricted": ["10"],
}


def _normalize_award_type(award_type: str) -> str:
    """"Cooperative Agreement", "cooperative agreement", and
    "cooperative_agreement" should all resolve the same way - the model
    isn't reliably going to reproduce the exact key format even when told
    what it is, the same lesson already learned from agency-name matching
    needing case-insensitive comparison."""
    return award_type.strip().lower().replace(" ", "_").replace("-", "_")


# USPS 2-letter codes for the 50 states + DC + the territories the live
# API's LocationObject.state field accepts (confirmed format from
# search_filters.md's examples, e.g. "state": "VA"). Code-owned because no
# live endpoint resolves a full state name to its abbreviation - the same
# "know the real finite vocabulary in code, don't trust the model to
# reproduce it" pattern as AWARD_TYPE_GROUPS.
US_STATE_ABBREVIATIONS: dict[str, str] = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
    "puerto rico": "PR", "guam": "GU", "american samoa": "AS",
    "u.s. virgin islands": "VI", "northern mariana islands": "MP",
}
_VALID_STATE_CODES = set(US_STATE_ABBREVIATIONS.values())


def _normalize_state(state: str) -> str:
    """Accepts a full state name ("Virginia") or a 2-letter USPS
    abbreviation ("VA"), case/spacing-insensitive, and returns the
    2-letter code LocationObject.state expects. Mirrors
    _normalize_award_type's normalize-then-look-up pattern: the model
    isn't reliably going to reproduce "VA" verbatim when the user said
    "Virginia," or vice versa.

    Raises USASpendingAPIError for anything unrecognized, rather than
    passing it through - the same silent-zero-results risk an unresolved
    agency name has.
    """
    normalized = state.strip().lower()
    if len(normalized) == 2 and normalized.upper() in _VALID_STATE_CODES:
        return normalized.upper()
    code = US_STATE_ABBREVIATIONS.get(normalized)
    if code is None:
        raise USASpendingAPIError(
            f"Unrecognized state '{state}'. Use a full state name (e.g. 'Virginia') "
            f"or a 2-letter USPS abbreviation (e.g. 'VA')."
        )
    return code


def _build_filters(
    client: USASpendingClient,
    agency_name: str,
    start_fiscal_year: int,
    end_fiscal_year: int,
    *,
    award_type: str | None = None,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
) -> AdvancedFilters:
    """Resolve agency_name + fiscal-year range into an AdvancedFilters -
    the shared first step of all three spending tools, replacing what was
    three near-identical blocks (agency resolution, date-range math,
    AdvancedFilters construction) duplicated across
    get_spending_by_category_raw, get_spending_over_time_raw, and
    search_awards_raw.

    Every parameter past end_fiscal_year is optional and defaults to
    None/unset. Omitting all of them reproduces exactly the original
    AdvancedFilters(agencies=..., time_period=...) shape - no
    award_type_codes, no recipient_search_text, no award_amounts, no
    location filters - so this is a pure consolidation for any caller
    that doesn't pass the new params, not a behavior change.

    client is an explicit parameter (not fetched internally via
    _get_usaspending_client()) so this stays unit-testable with a fake
    client.

    Raises USASpendingAPIError for every resolution failure, so the
    existing `except USASpendingAPIError` handling already in each
    @beta_tool wrapper covers all of these with no new except clause
    needed: agency_name doesn't resolve, award_type isn't a recognized
    AWARD_TYPE_GROUPS key, performed_in_state/recipient_in_state isn't a
    recognized state, or min_amount > max_amount (caught here instead of
    letting the live API reject an inverted bound with an opaque error).

    performed_in_state and recipient_in_state are genuinely different
    filters, not two spellings of the same thing: verified live that for
    DoD FY2023, performed_in_state="Virginia" gives $44.878B (place of
    performance) while recipient_in_state="Virginia" gives $60.46B
    (recipient address) - a ~$16B gap, verified live 2026-09-06. They map
    to different AdvancedFilters fields (place_of_performance_locations
    vs. recipient_locations) and can both be set at once.
    """
    agency = client.find_agency_by_name(agency_name)
    if agency is None:
        raise USASpendingAPIError(f"No agency found matching '{agency_name}'")

    start_date, end_date = fiscal_year_to_date_range(start_fiscal_year, end_fiscal_year)
    kwargs: dict = {
        "agencies": [AgencyFilter(type="awarding", tier="toptier", name=agency.agency_name)],
        "time_period": [TimePeriod(start_date=start_date, end_date=end_date)],
    }

    if award_type is not None:
        award_type_codes = AWARD_TYPE_GROUPS.get(_normalize_award_type(award_type))
        if award_type_codes is None:
            raise USASpendingAPIError(
                f"Unknown award_type '{award_type}'. Must be one of: {', '.join(AWARD_TYPE_GROUPS)}"
            )
        kwargs["award_type_codes"] = award_type_codes

    if recipient_name is not None:
        # API caps recipient_search_text at 1 item (search_filters.md).
        kwargs["recipient_search_text"] = [recipient_name]

    if min_amount is not None or max_amount is not None:
        if min_amount is not None and max_amount is not None and min_amount > max_amount:
            raise USASpendingAPIError(
                f"min_amount ({min_amount}) must not exceed max_amount ({max_amount})."
            )
        kwargs["award_amounts"] = [AwardAmount(lower_bound=min_amount, upper_bound=max_amount)]

    if performed_in_state is not None:
        kwargs["place_of_performance_locations"] = [
            LocationObject(state=_normalize_state(performed_in_state))
        ]

    if recipient_in_state is not None:
        kwargs["recipient_locations"] = [LocationObject(state=_normalize_state(recipient_in_state))]

    return AdvancedFilters(**kwargs)


# Fields valid across every award_type (per spending_by_award.md's "Base
# fields" list) - the amount field is NOT base, it's resolved per-award_type
# below and appended separately, since "Award Amount" is only valid for
# Contracts/IDVs/Non-Loan-Assistance - Loans expose "Loan Value" instead.
SEARCH_AWARDS_FIELDS_BASE = ["Award ID", "Recipient Name", "Awarding Agency", "Description"]

LOAN_AWARD_TYPE_CODES = {"07", "08"}


def _amount_field_for_award_type(award_type: str) -> str:
    """"Award Amount" is valid for Contracts, IDVs, and Non-Loan
    Assistance award types per spending_by_award.md's field tables, but
    Loans (codes 07/08) expose "Loan Value" instead - sorting or reading
    "Award Amount" for a loan-type search would be invalid/empty for that
    field. IDV codes are never in AWARD_TYPE_GROUPS (see its docstring),
    so loans-vs-everything-else is the only branch this codebase needs.

    Live-verified 2026-09-06 (dev_tools/verify_shared_filters.py): "Loan
    Value" is a real field on live loan-type search_awards results, and
    sorting by it doesn't error.

    Assumes award_type is already a valid AWARD_TYPE_GROUPS key - callers
    only reach this after _build_filters has already validated it earlier
    in the same call, so an unrecognized key here falls back to "Award
    Amount" rather than raising a second time.
    """
    codes = AWARD_TYPE_GROUPS.get(_normalize_award_type(award_type), [])
    return "Loan Value" if any(c in LOAN_AWARD_TYPE_CODES for c in codes) else "Award Amount"


def _record_optional_filter_context(
    context: dict,
    *,
    award_type: str | None = None,
    recipient_name: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
) -> dict:
    """Adds each of the six optional filter params to a citation context
    dict, but only the ones actually set - so a citation reflects exactly
    which filters were used for that call, not every filter this tool
    supports in the abstract."""
    for key, value in (
        ("award_type", award_type),
        ("recipient_name", recipient_name),
        ("min_amount", min_amount),
        ("max_amount", max_amount),
        ("performed_in_state", performed_in_state),
        ("recipient_in_state", recipient_in_state),
    ):
        if value is not None:
            context[key] = value
    return context

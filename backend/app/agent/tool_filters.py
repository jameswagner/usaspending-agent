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

import re
from typing import Literal

from backend.app.usaspending_client import (
    AdvancedFilters,
    AgencyFilter,
    AwardAmount,
    LocationObject,
    NAICSCodeObject,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

from .recipient_types import _normalize_recipient_type
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


# A static Literal mirror of AWARD_TYPE_GROUPS.keys(), used as the actual
# type hint on every @beta_tool award_type parameter so beta_tool's schema
# generation emits a real JSON-schema `enum` - a hard constraint on what
# the model can generate, not just a runtime check after the fact (see
# TestLiteralTypesMatchVocabulary in tests/test_agent.py, which asserts
# this can't silently drift from AWARD_TYPE_GROUPS). Written statically
# rather than derived (e.g. Literal[*AWARD_TYPE_GROUPS] via unpacking)
# to avoid any uncertainty about dynamic Literal construction interacting
# with `from __future__ import annotations`-deferred evaluation - verified
# live that beta_tool correctly resolves a plain static Literal into a
# real enum; the test is what actually keeps this in sync, not cleverness
# in how it's built.
AwardType = Literal[
    "contracts", "grants", "loans",
    "bpa_call", "purchase_order", "delivery_order", "definitive_contract",
    "direct_loan", "guaranteed_loan",
    "block_grant", "formula_grant", "project_grant", "cooperative_agreement",
    "insurance", "other_financial_assistance",
    "direct_payment_specified", "direct_payment_unrestricted",
]


def _normalize_award_type(award_type: str) -> str:
    """"Cooperative Agreement", "cooperative agreement", and
    "cooperative_agreement" should all resolve the same way - the model
    isn't reliably going to reproduce the exact key format even when told
    what it is, the same lesson already learned from agency-name matching
    needing case-insensitive comparison."""
    return award_type.strip().lower().replace(" ", "_").replace("-", "_")


# The 7 non-overlapping "leaf" categories that together cover every
# possible award (IDVs excluded - see AWARD_TYPE_GROUPS's docstring). The
# other 10 AWARD_TYPE_GROUPS keys are sub-types whose codes are already a
# subset of one of these seven's codes (e.g. "contracts" = A,B,C,D, the
# same codes "bpa_call"/"purchase_order"/"delivery_order"/
# "definitive_contract" split out individually) - so once a broad
# bucket's search comes back empty, trying its own sub-type afterward is
# guaranteed to also come back empty, not a fresh thing to check.
EXHAUSTIVE_AWARD_TYPE_CATEGORIES = [
    "contracts", "grants", "loans", "insurance",
    "other_financial_assistance", "direct_payment_specified", "direct_payment_unrestricted",
]

_BROAD_CATEGORY_FOR_AWARD_TYPE = {
    "contracts": "contracts", "bpa_call": "contracts", "purchase_order": "contracts",
    "delivery_order": "contracts", "definitive_contract": "contracts",
    "grants": "grants", "block_grant": "grants", "formula_grant": "grants",
    "project_grant": "grants", "cooperative_agreement": "grants",
    "loans": "loans", "direct_loan": "loans", "guaranteed_loan": "loans",
    "insurance": "insurance",
    "other_financial_assistance": "other_financial_assistance",
    "direct_payment_specified": "direct_payment_specified",
    "direct_payment_unrestricted": "direct_payment_unrestricted",
}


# Built from AWARD_TYPE_GROUPS.keys() so it can't drift from the real
# vocabulary: every group name with underscores spaced out, plus its
# naive singular/plural counterpart ("grants"/"grant", "insurance"/
# "insurances"). Used only to catch a `keywords` value that just restates
# the award-type/category filter itself - see _keywords_restates_award_type.
_AWARD_TYPE_KEYWORD_TERMS: set[str] = set()
for _award_type_key in AWARD_TYPE_GROUPS:
    _spaced = _award_type_key.replace("_", " ")
    _AWARD_TYPE_KEYWORD_TERMS.add(_spaced)
    _AWARD_TYPE_KEYWORD_TERMS.add(_spaced[:-1] if _spaced.endswith("s") else _spaced + "s")
del _award_type_key, _spaced


def _keywords_restates_award_type(keywords: str) -> bool:
    """True if `keywords`, once normalized, is nothing but an award-type/
    category term (e.g. "grant", "Contracts", "cooperative-agreement") -
    the exact #124 bug: award_type already scopes results precisely, so
    restating it as a keyword only narrows results to awards whose
    description text happens to contain that literal word, which most
    awards of that type don't (most grant descriptions never say
    "grant"). A real topic phrase within the category (e.g. "grants for
    flood mitigation") normalizes to something other than a bare term
    here and is left alone.
    """
    normalized = re.sub(r"[\s_-]+", " ", keywords.strip().lower())
    return normalized in _AWARD_TYPE_KEYWORD_TERMS


def _other_award_type_categories_to_try(award_type: str) -> str:
    """Comma-joined list of the exhaustive categories not already implied
    by award_type - for a zero-results message, so the model has a
    concrete next step instead of concluding no award records exist after
    one or two guesses. Found live (2026-09-11): asked for the top awards
    under a Social Security retirement CFDA program, the model tried
    "grants" then "contracts" (both genuinely zero - the real records are
    direct_payment_unrestricted), got zero both times, and fabricated an
    explanation ("these payments aren't tracked as individual awards")
    instead of trying the remaining categories - live search confirmed
    1,738 real award records existed the whole time.
    """
    current_broad = _BROAD_CATEGORY_FOR_AWARD_TYPE[_normalize_award_type(award_type)]
    return ", ".join(c for c in EXHAUSTIVE_AWARD_TYPE_CATEGORIES if c != current_broad)


# POST /api/v2/recipient/'s own award_type enum (recipient.md) - a real,
# different, coarser vocabulary from AWARD_TYPE_GROUPS/AwardType above, not
# reusable: only 6 broad buckets, no sub-type granularity (no
# cooperative_agreement, no bpa_call), and "direct_payments" here is
# singular where AwardType splits it into direct_payment_specified/
# direct_payment_unrestricted. Confirmed from the live contract, not
# assumed to line up just because both are "award type" filters.
RECIPIENT_AWARD_TYPES = {
    "all", "contracts", "grants", "loans", "direct_payments", "other_financial_assistance",
}

RecipientAwardType = Literal[
    "all", "contracts", "grants", "loans", "direct_payments", "other_financial_assistance",
]


def _normalize_recipient_award_type(award_type: str) -> str:
    normalized = award_type.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in RECIPIENT_AWARD_TYPES:
        raise USASpendingAPIError(
            f"Unknown award_type '{award_type}'. Must be one of: {', '.join(sorted(RECIPIENT_AWARD_TYPES))}"
        )
    return normalized


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


def _normalize_county_fips(county: str) -> str:
    """3-digit zero-padded county FIPS code, e.g. "025" for Yavapai County,
    AZ - confirmed live an unpadded "25" silently returns zero results
    (no error), so this is enforced here rather than left to the model.
    Also accepts a 5-digit state+county FIPS (resolve_county_fips's own
    county_fips field) by taking the last 3 digits - confirmed live the
    filter wants county-only, not the state-prefixed form.
    """
    digits = re.sub(r"\D", "", county)
    if len(digits) == 5:
        digits = digits[-3:]
    if not digits or len(digits) > 3:
        raise USASpendingAPIError(
            f"'{county}' doesn't look like a county FIPS code (expected 3 digits, e.g. '025' for "
            "Yavapai County, AZ - use resolve_county_fips to find one from a county name)."
        )
    return digits.zfill(3)


def _normalize_district(district: str) -> str:
    """2-digit zero-padded congressional district number, e.g. "01" - also
    accepts the "AZ-01" shape (autocomplete/location's own current_cd/
    original_cd field) by keeping only the digits. Confirmed live the
    filter rejects both the unpadded ("1") and "ST-NN" forms outright with
    a 400, unlike county's silent failure - still normalized here rather
    than relying on the model to hit and recover from that error.
    """
    digits = re.sub(r"\D", "", district)
    if not digits or len(digits) > 2:
        raise USASpendingAPIError(
            f"'{district}' doesn't look like a congressional district number (expected e.g. '01', "
            "or 'AZ-01')."
        )
    return digits.zfill(2)


def _build_location(
    state: str | None,
    county: str | None,
    city: str | None,
    zip_code: str | None,
    district: str | None,
) -> LocationObject | None:
    """Combines one place's state/county/city/zip/district into a single
    LocationObject, enforcing the two real constraints search_filters.md's
    StandardLocationObject spec states outright: county requires state,
    and county/district can't both be set on one location entry.
    """
    if all(v is None for v in (state, county, city, zip_code, district)):
        return None

    if county is not None and state is None:
        raise USASpendingAPIError(
            "A county filter also requires a state - county names/FIPS codes repeat across states."
        )
    if county is not None and district is not None:
        raise USASpendingAPIError(
            "county and district can't both be set on the same location - the live API forbids "
            "combining them."
        )

    kwargs: dict = {}
    if state is not None:
        kwargs["state"] = _normalize_state(state)
    if county is not None:
        kwargs["county"] = _normalize_county_fips(county)
    if city is not None:
        kwargs["city"] = city.strip()
    if zip_code is not None:
        kwargs["zip"] = zip_code.strip()
    if district is not None:
        kwargs["district_current"] = _normalize_district(district)
    return LocationObject(**kwargs)


# The API's own real values (search_filters.md's Award/Transaction Search
# Time Period Objects), not guessed. action_date is the API's own default
# when omitted - kept in this set so an explicit "action_date" passed by
# the model still validates instead of erroring on a value that's
# actually correct.
VALID_DATE_TYPES = {"action_date", "date_signed", "last_modified_date", "new_awards_only"}

# Static Literal mirror of VALID_DATE_TYPES - same reasoning as AwardType above.
DateType = Literal["action_date", "date_signed", "last_modified_date", "new_awards_only"]


def _normalize_date_type(date_type: str) -> str:
    """Case/spacing-insensitive lookup against VALID_DATE_TYPES, same
    normalize-then-validate pattern as _normalize_award_type/_normalize_state.
    """
    normalized = date_type.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in VALID_DATE_TYPES:
        raise USASpendingAPIError(
            f"Unrecognized date_type '{date_type}'. Must be one of: {', '.join(sorted(VALID_DATE_TYPES))}"
        )
    return normalized


# Static Literal mirror of the domestic/foreign vocabulary - same
# reasoning as AwardType above.
Scope = Literal["domestic", "foreign"]


def _normalize_scope(scope: str, param_name: str) -> str:
    """domestic/foreign - the only two values place_of_performance_scope
    and recipient_scope accept (search_filters.md)."""
    normalized = scope.strip().lower()
    if normalized not in ("domestic", "foreign"):
        raise USASpendingAPIError(f"Unrecognized {param_name} '{scope}'. Must be 'domestic' or 'foreign'.")
    return normalized


# spending_by_geography's own scope/geo_layer enums - unrelated to Scope
# above (domestic/foreign); this is "which location field to aggregate on"
# and "what granularity to group by".
GeoScope = Literal["place_of_performance", "recipient_location"]
GeoLayer = Literal["state", "county", "district", "country"]


# Format-only validation for the three code passthroughs below - these are
# large government classification systems (thousands of NAICS/PSC codes,
# hundreds of CFDA programs) with no small closed vocabulary to validate
# existence against the way AWARD_TYPE_GROUPS/US_STATE_ABBREVIATIONS do.
# Catching an obviously-malformed value (the model passing a description
# instead of a code) is what's actually achievable here; a real invalid-
# but-well-formed code still just gets a live 400/empty-results from the
# API itself, the same graceful-decline shape as an unknown category.
def _validate_naics_code(naics_code: str) -> str:
    code = naics_code.strip()
    if not code.isdigit() or not (2 <= len(code) <= 6):
        raise USASpendingAPIError(
            f"'{naics_code}' doesn't look like a NAICS code (expected 2-6 digits, e.g. '541511'). "
            "If you don't have the exact code, ask for a category breakdown by naics instead."
        )
    return code


def _validate_psc_code(psc_code: str) -> str:
    code = psc_code.strip().upper()
    if not (len(code) == 4 and code.isalnum()):
        raise USASpendingAPIError(
            f"'{psc_code}' doesn't look like a PSC code (expected a 4-character code, e.g. '7030')."
        )
    return code


_CFDA_PATTERN = re.compile(r"^\d{2}\.\d{3}$")


def _validate_cfda_program(cfda_program: str) -> str:
    code = cfda_program.strip()
    if not _CFDA_PATTERN.match(code):
        raise USASpendingAPIError(
            f"'{cfda_program}' doesn't look like a CFDA/assistance listing number (expected NN.NNN, e.g. '10.001')."
        )
    return code


def _build_filters(
    client: USASpendingClient,
    agency_name: str | None,
    start_fiscal_year: int,
    end_fiscal_year: int,
    *,
    award_type: str | None = None,
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
    recognized state, min_amount > max_amount (caught here instead of
    letting the live API reject an inverted bound with an opaque error),
    or keywords is nothing but an award-type/category term restating
    award_type itself (see _keywords_restates_award_type - the #124 bug:
    that combination doesn't error at the live API, it silently narrows
    results instead, which is worse, so it's rejected here up front).

    performed_in_state and recipient_in_state are genuinely different
    filters, not two spellings of the same thing: verified live that for
    DoD FY2023, performed_in_state="Virginia" gives $44.878B (place of
    performance) while recipient_in_state="Virginia" gives $60.46B
    (recipient address) - a ~$16B gap, verified live 2026-09-06. They map
    to different AdvancedFilters fields (place_of_performance_locations
    vs. recipient_locations) and can both be set at once.

    naics_code/psc_code/cfda_program are direct code passthroughs (format-
    validated only, not looked up against a vocabulary) - unlike
    award_type/state, these codes are already the exact value the API
    wants once an analyst states them (e.g. "NAICS 541511"), the same way
    a stock ticker doesn't need translating. No mapping table exists for
    them the way AWARD_TYPE_GROUPS/US_STATE_ABBREVIATIONS do, because
    there's no small closed vocabulary to hardcode - NAICS/PSC/CFDA are
    each thousands of entries. A keyword->code lookup (the live
    autocomplete/naics/psc/cfda endpoints) was considered and deliberately
    not built - see ADVANCED_FILTER_FIELD_COVERAGE's naics_codes entry.

    agency_name is optional - a recipient-only, cross-agency question needs
    to work too. At least one of agency_name/recipient_name/recipient_id/
    performed_in_state/recipient_in_state/naics_code/psc_code/cfda_program/
    keywords/award_id/description must be given, or this raises;
    award_type/min_amount/max_amount/date_type/*_scope/recipient_type
    don't count on their own (see #16) - recipient_type is a broad
    classification (e.g. "small_business" alone still spans nearly all of
    federal spending), the same reasoning that excludes award_type.

    recipient_id is a real, precise filter - confirmed live 2026-09-08 to
    reproduce a recipient's true all-time total to the penny, unlike
    recipient_name (a text match, confirmed wrong in both directions: it
    can sweep in unrelated similarly-named entities, e.g. a joint venture,
    AND miss real subsidiaries whose legal name doesn't contain the
    parent's name at all - see BACKLOG.md's recipient-profile entry).
    Prefer recipient_id whenever one has already been resolved (e.g. via
    search_recipients). Only wired through here for
    get_spending_by_category/get_spending_over_time - confirmed live that
    search_awards silently ignores this filter entirely (the live API's
    own `messages` field says so explicitly), so it's never passed through
    on that tool's path.

    award_id/recipient_type/description are plain passthroughs into
    award_ids/recipient_type_names/description (all live-verified
    2026-09-12 against the real API, across search_awards,
    get_spending_by_category, and get_spending_by_geography). description
    is distinct from keywords - description does a phrase-prefix match
    against the award's own description text only, where keywords also
    matches PIID/FAIN/URI and several other text fields (recipient name,
    NAICS/PSC description, etc.) - so a keywords hit doesn't imply a
    description hit or vice versa.
    """
    real_scoping_filters = (
        agency_name, recipient_name, recipient_id,
        performed_in_state, recipient_in_state,
        performed_in_county, recipient_in_county,
        performed_in_city, recipient_in_city,
        performed_in_zip, recipient_in_zip,
        performed_in_district, recipient_in_district,
        naics_code, psc_code, cfda_program, keywords,
        award_id, description,
    )
    if all(f is None for f in real_scoping_filters):
        raise USASpendingAPIError(
            "At least one of agency_name, recipient_name, recipient_id, performed_in_state, "
            "recipient_in_state, performed_in_county, recipient_in_county, performed_in_city, "
            "recipient_in_city, performed_in_zip, recipient_in_zip, performed_in_district, "
            "recipient_in_district, naics_code, psc_code, cfda_program, keywords, award_id, or "
            "description must be given - a question scoped by none of them would mean all federal "
            "spending, ever."
        )

    start_date, end_date = fiscal_year_to_date_range(start_fiscal_year, end_fiscal_year)
    kwargs: dict = {
        "time_period": [TimePeriod(start_date=start_date, end_date=end_date)],
    }

    if agency_name is not None:
        agency = client.find_agency_by_name(agency_name)
        if agency is None:
            raise USASpendingAPIError(f"No agency found matching '{agency_name}'")
        kwargs["agencies"] = [AgencyFilter(type="awarding", tier="toptier", name=agency.agency_name)]

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

    if recipient_id is not None:
        kwargs["recipient_id"] = recipient_id

    if min_amount is not None or max_amount is not None:
        if min_amount is not None and max_amount is not None and min_amount > max_amount:
            raise USASpendingAPIError(
                f"min_amount ({min_amount}) must not exceed max_amount ({max_amount})."
            )
        kwargs["award_amounts"] = [AwardAmount(lower_bound=min_amount, upper_bound=max_amount)]

    performed_location = _build_location(
        performed_in_state, performed_in_county, performed_in_city, performed_in_zip, performed_in_district
    )
    if performed_location is not None:
        kwargs["place_of_performance_locations"] = [performed_location]

    recipient_location = _build_location(
        recipient_in_state, recipient_in_county, recipient_in_city, recipient_in_zip, recipient_in_district
    )
    if recipient_location is not None:
        kwargs["recipient_locations"] = [recipient_location]

    if keywords is not None:
        if _keywords_restates_award_type(keywords):
            raise USASpendingAPIError(
                f"keywords='{keywords}' just restates an award-type/category term. award_type "
                "already scopes results to that category precisely - restating it as a keyword "
                "on top silently narrows results to only the awards whose description text "
                "happens to contain that exact word, which most awards of that type don't (e.g. "
                "most grant descriptions never say \"grant\"). Use award_type for the category, "
                "and keywords only for a topic within it (e.g. 'flood mitigation'), or omit "
                "keywords entirely."
            )
        kwargs["keywords"] = [keywords]

    if date_type is not None:
        kwargs["time_period"][0].date_type = _normalize_date_type(date_type)

    if place_of_performance_scope is not None:
        kwargs["place_of_performance_scope"] = _normalize_scope(
            place_of_performance_scope, "place_of_performance_scope"
        )

    if recipient_scope is not None:
        kwargs["recipient_scope"] = _normalize_scope(recipient_scope, "recipient_scope")

    if naics_code is not None:
        kwargs["naics_codes"] = NAICSCodeObject(require=[_validate_naics_code(naics_code)])

    if psc_code is not None:
        # Flat list form, not the hierarchical require/exclude object -
        # verified live 2026-09-06 that a plain psc_codes: [code] list
        # correctly filters by that exact code.
        kwargs["psc_codes"] = [_validate_psc_code(psc_code)]

    if cfda_program is not None:
        kwargs["program_numbers"] = [_validate_cfda_program(cfda_program)]

    if award_id is not None:
        kwargs["award_ids"] = [award_id.strip()]

    if recipient_type is not None:
        kwargs["recipient_type_names"] = [_normalize_recipient_type(recipient_type)]

    if description is not None:
        kwargs["description"] = description

    return AdvancedFilters(**kwargs)


# Fields valid across every award_type (per spending_by_award.md's "Base
# fields" list) - the amount field is NOT base, it's resolved per-award_type
# below and appended separately, since "Award Amount" is only valid for
# Contracts/IDVs/Non-Loan-Assistance - Loans expose "Loan Value" instead.
#
# generated_internal_id is the hash-style id (e.g.
# "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-") that GET /api/v2/awards/
# {award_id}/ actually requires - confirmed live (2026-09-08) that the
# plain "Award ID" (PIID/FAIN) 404s there. Live-verified present and
# non-empty across 993 real awards spanning every award_type family
# (contracts A/B/C/D, all 8 IDV sub-types, grants, loans, direct
# payments, insurance/other) across 12 agencies - safe to rely on
# unconditionally, not just for the common cases.
SEARCH_AWARDS_FIELDS_BASE = ["Award ID", "generated_internal_id", "Recipient Name", "Awarding Agency", "Description"]

# Common to both Contract Subawards and Grant Subawards field lists
# (spending_by_award.md) - skips the type-specific extras (NAICS/PSC for
# contracts, Assistance Listing for grants) rather than branching per
# award_type, matching SEARCH_AWARDS_FIELDS_BASE's own base-fields approach.
SUBAWARD_FIELDS = [
    "Sub-Award ID",
    "Sub-Awardee Name",
    "Sub-Award Amount",
    "Sub-Award Date",
    "Sub-Award Description",
    "Prime Award ID",
    "Prime Recipient Name",
    "Awarding Agency",
    "prime_award_generated_internal_id",
]

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
    agency_name: str | None = None,
    award_type: str | None = None,
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
) -> dict:
    """Adds each optional filter param to a citation context dict, but
    only the ones actually set - so a citation reflects exactly which
    filters were used for that call, not every filter this tool supports
    in the abstract."""
    for key, value in (
        ("agency_name", agency_name),
        ("award_type", award_type),
        ("recipient_name", recipient_name),
        ("recipient_id", recipient_id),
        ("min_amount", min_amount),
        ("max_amount", max_amount),
        ("performed_in_state", performed_in_state),
        ("recipient_in_state", recipient_in_state),
        ("performed_in_county", performed_in_county),
        ("recipient_in_county", recipient_in_county),
        ("performed_in_city", performed_in_city),
        ("recipient_in_city", recipient_in_city),
        ("performed_in_zip", performed_in_zip),
        ("recipient_in_zip", recipient_in_zip),
        ("performed_in_district", performed_in_district),
        ("recipient_in_district", recipient_in_district),
        ("keywords", keywords),
        ("date_type", date_type),
        ("place_of_performance_scope", place_of_performance_scope),
        ("recipient_scope", recipient_scope),
        ("naics_code", naics_code),
        ("psc_code", psc_code),
        ("cfda_program", cfda_program),
        ("award_id", award_id),
        ("recipient_type", recipient_type),
        ("description", description),
    ):
        if value is not None:
            context[key] = value
    return context


# The live API's own hard ceiling - verified live (BACKLOG.md "Red team:
# resource abuse"): limit=1000 got a clean 422 "above max 100". Clamping
# silently rather than raising is safe now specifically because
# page_metadata.hasNext + tools.py's _truncation_note already tell the
# model honestly when a clamped result set isn't exhaustive - clamping
# doesn't reopen the "confidently incomplete" problem that fix closed,
# it just avoids a confusing 422 for a value the model was always going
# to get an incomplete-but-honest answer for anyway.
MAX_LIMIT = 100


def _clamp_limit(limit: int) -> int:
    return max(1, min(limit, MAX_LIMIT))

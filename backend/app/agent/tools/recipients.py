"""search_recipients / get_recipient_details - resolving a recipient name
to an exact recipient_id, and getting that recipient's own profile. Not
_build_filters-based - these call the client's recipient endpoints directly.
"""
from __future__ import annotations

import logging

from anthropic import beta_tool

from backend.app.usaspending_client import (
    RecipientListing,
    RecipientLocation,
    RecipientOverview,
    USASpendingAPIError,
)

from ..singletons import _get_usaspending_client
from ..tool_filters import (
    RecipientAwardType,
    _clamp_limit,
    _normalize_recipient_award_type,
)
from ._shared import (
    _check_tool_call_budget,
    _record_tool_call,
    _truncation_note,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)


def _format_recipient_level(level: str) -> str:
    return {"P": "parent", "C": "child", "R": "standalone"}.get(level, level)


# Full recipientTypes map from usaspending-website's own recipientType.js, not a
# hand-sampled subset - this is what usaspending.gov itself renders for every code.
BUSINESS_TYPE_LABELS: dict[str, str] = {
    "category_business": "Business",
    "business": "Business",
    "small_business": "Small Business",
    "other_than_small_business": "Other Than Small Business",
    "corporate_entity_tax_exempt": "Corporate Entity Tax Exempt",
    "corporate_entity_not_tax_exempt": "Corporate Entity Not Tax Exempt",
    "partnership_or_limited_liability_partnership": "Partnership or Limited Liability Partnership",
    "sole_proprietorship": "Sole Proprietorship",
    "manufacturer_of_goods": "Manufacturer of Goods",
    "subchapter_s_corporation": "Sub-Chapter S Corporation",
    "limited_liability_corporation": "Limited Liability Corporation (LLC)",
    "category_minority_owned_business": "Minority Owned Business",
    "minority_owned_business": "Minority Owned Business",
    "alaskan_native_corporation_owned_firm": "Alaskan Native Corporation Owned Firm",
    "american_indian_owned_business": "American Indian Owned Business",
    "asian_pacific_american_owned_business": "Asian Pacific American Owned Business",
    "black_american_owned_business": "Black American Owned Business",
    "hispanic_american_owned_business": "Hispanic American Owned Business",
    "native_american_owned_business": "Native American Owned Business",
    "native_hawaiian_organization_owned_firm": "Native Hawaiian Organization Owned Firm",
    "subcontinent_asian_indian_american_owned_business": "Subcontinent Asian Indian American Owned Business",
    "tribally_owned_firm": "Tribally Owned Firm",
    "other_minority_owned_business": "Other Minority Owned Business",
    "woman_owned_business": "Women Owned Business",
    "category_women_owned_small_business": "Women Owned Small Business",
    "women_owned_small_business": "Women Owned Small Business",
    "economically_disadvantaged_women_owned_small_business": "Economically Disadvantaged Women Owned Small Business",
    "joint_venture_women_owned_small_business": "Joint Venture Women Owned Small Business",
    "joint_venture_economically_disadvantaged_women_owned_small_business": "Joint Venture Economically Disadvantaged Women Owned Small Business",
    "category_veteran_owned_business": "Veteran Owned Business",
    "veteran_owned_business": "Veteran Owned Business",
    "service_disabled_veteran_owned_business": "Service Disabled Veteran Owned Business",
    "category_special_designations": "Special Designations",
    "special_designations": "Special Designations",
    "8a_program_participant": "8a Program Participant",
    "ability_one_program": "Ability One Program",
    "dot_certified_disadvantaged_business_enterprise": "DoT Certified Disadvantaged Business Enterprise",
    "emerging_small_business": "Emerging Small Business",
    "federally_funded_research_and_development_corp": "Federally Funded Research and Development Corp",
    "historically_underutilized_business_firm": "Historically Underutilized Business (HUBZone) Firm",
    "labor_surplus_area_firm": "Labor Surplus Area Firm",
    "sba_certified_8a_joint_venture": "SBA Certified 8a Joint Venture",
    "self_certified_small_disadvanted_business": "Self-Certified Small Disadvantaged Business",
    "small_agricultural_cooperative": "Small Agricultural Cooperative",
    "community_developed_corporation_owned_firm": "Community Developed Corporation Owned Firm",
    "us_owned_business": "U.S. Owned Business",
    "foreign_owned_and_us_located_business": "Foreign-Owned and U.S. Located Business",
    "foreign_owned": "Foreign Owned",
    "foreign_government": "Foreign Government",
    "international_organization": "International Organization",
    "domestic_shelter": "Domestic Shelter",
    "hospital": "Hospital",
    "veterinary_hospital": "Veterinary Hospital",
    "category_nonprofit": "Nonprofit",
    "nonprofit": "Nonprofit",
    "foundation": "Foundation",
    "community_development_corporations": "Community Development Corporations",
    "category_higher_education": "Higher Education",
    "higher_education": "Higher Education",
    "public_institution_of_higher_education": "Public Institution of Higher Education",
    "private_institution_of_higher_education": "Private Institution of Higher Education",
    "minority_serving_institution_of_higher_education": "Minority-Serving Institution of Higher Education",
    "school_of_forestry": "School of Forestry",
    "veterinary_college": "Veterinary College",
    "category_government": "Government",
    "government": "Government",
    "national_government": "National Government",
    "interstate_entity": "Interstate Entity",
    "regional_and_state_government": "Regional and State Government",
    "regional_organization": "Regional Organization",
    "us_territory_or_possession": "U.S. Territory or Possession",
    "council_of_governments": "Council of Governments",
    "local_government": "Local Government",
    "indian_native_american_tribal_government": "Indian Native American Tribal Government",
    "authorities_and_commissions": "Authorities and Commissions",
    "category_individuals": "Individuals",
    "individuals": "Individuals",
}


def _format_business_type(code: str) -> str:
    return BUSINESS_TYPE_LABELS.get(code, code.replace("_", " ").title())


def _format_recipient_listing(listing: RecipientListing) -> str:
    """One line per search_recipients candidate. amount is always
    trailing-12-months (RecipientListing.amount's own docstring) - labeled
    explicitly so it's never mistaken for the all-time total
    get_recipient_details can give instead."""
    ids = [f"UEI {listing.uei}" if listing.uei else None, f"DUNS {listing.duns}" if listing.duns else None]
    id_str = f" ({', '.join(i for i in ids if i)})" if any(ids) else ""
    return (
        f"{listing.name or 'unknown'} [{_format_recipient_level(listing.recipient_level)}]{id_str} - "
        f"${listing.amount:,.2f} (last 12 months) [recipient_id: {listing.id}]"
    )


def _format_recipient_address(location: RecipientLocation | None) -> str:
    """Full street address - the real usaspending.gov recipient page shows
    this in full for a normal business, unlike the award-side recipient/
    place-of-performance trim (state/city only), which exists for a
    different reason (redacting an individual's home address). Callers use
    _format_recipient_state_only instead of this specifically for the
    redacted/aggregate bucket case - see _format_recipient_overview."""
    if not location:
        return "N/A"
    street = ", ".join(p for p in [location.address_line1, location.address_line2, location.address_line3] if p)
    city_state_zip = " ".join(p for p in [location.city_name, location.state_code, location.zip] if p)
    parts = [p for p in [street, city_state_zip, location.country_name] if p]
    return ", ".join(parts) if parts else "N/A"


def _format_recipient_state_only(location: RecipientLocation | None) -> str:
    if not location:
        return "N/A"
    return location.state_code or location.country_name or "N/A"


# The live sentinel string for a pooled bucket of PII-redacted individual
# recipients (RecipientOverview has no typed flag for this, unlike the
# award side's record_type - only this literal name string). A real
# example totals $14.9B across 2.24M transactions - clearly not one
# person. Never shown verbatim to the model/user as if it were a real
# name.
_REDACTED_RECIPIENT_NAME = "REDACTED DUE TO PII"


def _format_recipient_overview(overview: RecipientOverview) -> str:
    is_redacted = overview.name == _REDACTED_RECIPIENT_NAME
    lines: list[str] = []

    if is_redacted:
        lines.append(
            "This recipient_id represents a pooled aggregate of many PII-redacted individual "
            "recipients, not one person or entity - the totals below are NOT one recipient's "
            "spending. (Real example confirmed live: $14.9B across 2.24M transactions.)"
        )
    else:
        lines.append(overview.name or "unknown")
        if overview.alternate_names:
            lines.append(f"Also known as: {', '.join(overview.alternate_names)}")

    lines.append(f"Recipient level: {_format_recipient_level(overview.recipient_level)}")
    if overview.uei:
        lines.append(f"UEI: {overview.uei}")
    if overview.duns:
        lines.append(f"Legacy DUNS: {overview.duns}")

    if overview.parent_id and overview.parent_id != overview.recipient_id:
        lines.append(f"Parent: {overview.parent_name or 'unknown'} [recipient_id: {overview.parent_id}]")

    location_label = _format_recipient_state_only(overview.location) if is_redacted else _format_recipient_address(overview.location)
    lines.append(f"Location: {location_label}")

    if overview.business_types:
        readable = ", ".join(_format_business_type(bt) for bt in overview.business_types)
        lines.append(f"Business types: {readable}")

    lines.append(
        f"Total: ${overview.total_transaction_amount:,.2f} across {overview.total_transactions:,} transactions"
    )
    # Always shown, not suppressed at zero, unlike get_award_details's loan
    # case - the real usaspending.gov page always shows this line ("$0
    # from 0 transactions"), and there's no adjacent nonzero figure here to
    # make a zero read as contradictory the way it did for a guaranteed
    # loan.
    lines.append(
        f"Face value of loans: ${overview.total_face_value_loan_amount:,.2f} across "
        f"{overview.total_face_value_loan_transactions:,} transactions"
    )

    return "\n".join(lines)


@beta_tool
def search_recipients(keyword: str | None = None, award_type: RecipientAwardType = "all", limit: int = 10) -> str:
    """Search for a recipient (company, organization, or individual) by name, UEI, or DUNS number, to find its exact recipient_id for a precise follow-up query (get_recipient_details, or the recipient_id parameter on get_spending_by_category/get_spending_over_time). Use this whenever a question names a specific real recipient — do not guess a recipient_id, and prefer this over a bare recipient_name text filter whenever precision matters.

    A plain company name is genuinely ambiguous at this scale — confirmed live that "Leidos" and "Boeing" each resolve to 6+ distinct recipient_ids sharing the exact same display name (parent companies, subsidiaries, and historical registrations from mergers/acquisitions). This tool shows every real candidate rather than silently picking one. If several results share a name, prefer the one with recipient level "parent" for a "how much has this company received in total" question — confirmed live to be a true, complete rollup across all of that company's own child registrations, to the penny. Ask the user to disambiguate if it's still unclear which candidate they mean.

    An exact UEI or DUNS as the keyword returns a single, precise match (confirmed live) — use one directly if you already have it.

    Args:
        keyword: Optional. A recipient's name, UEI, or DUNS number, e.g. "Boeing" or
            "NU2UC8MX6NK1". Omit for an unscoped, globally-ranked top-recipients list.
        award_type: Optional. Restrict to one broad award-type bucket — all (default),
            contracts, grants, loans, direct_payments, or other_financial_assistance. A
            different, coarser vocabulary than every other tool's award_type parameter here —
            no sub-type granularity (no cooperative_agreement, no bpa_call).
        limit: Max number of candidates to return (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    limit = _clamp_limit(limit)
    try:
        normalized_award_type = _normalize_recipient_award_type(award_type)
        client = _get_usaspending_client()
        response = client.search_recipients(keyword, award_type=normalized_award_type, limit=limit)
    except USASpendingAPIError as e:
        logger.warning("search_recipients failed for %r: %s", keyword, e)
        return f"This query failed: {e}."

    _record_tool_call("search_recipients", response, {"keyword": keyword})

    if not response.results:
        return f"No recipients found matching '{keyword}'." if keyword else "No recipients found."

    lines = [_format_recipient_listing(r) for r in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
    return _wrap_untrusted("\n".join(lines) + note)


@beta_tool
def get_recipient_details(recipient_id: str, year: str = "all") -> str:
    """Get full profile details for one specific, already-resolved recipient: identity (name, alternate names, UEI/Legacy DUNS), parent relationship, address, business types, and total federal transaction amount for the given time period. Use this as a follow-up after search_recipients has resolved a specific recipient_id — not for browsing or searching by name, which search_recipients already does.

    year defaults to "all" (the recipient's entire history), not the live API's own default
    of "latest" (trailing 12 months) — "latest" would just repeat the same number
    search_recipients already showed for the same candidate (confirmed live: search_recipients's
    own amount is always trailing-12-months and never respects year), so defaulting here to
    "all" gives a genuinely different, additive answer instead of restating one.

    Args:
        recipient_id: The exact recipient_id from a prior search_recipients result (e.g.
            "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"). Do not guess or construct one.
        year: A specific fiscal year (e.g. "2023"), "all" (default — the recipient's entire
            history), or "latest" (trailing 12 months — the same window search_recipients
            already shows, so rarely useful here unless explicitly asked for).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        client = _get_usaspending_client()
        overview = client.get_recipient(recipient_id, year=year)
    except USASpendingAPIError as e:
        logger.warning("get_recipient_details failed for %s: %s", recipient_id, e)
        return f"This query failed: {e}."

    _record_tool_call("get_recipient_details", overview, {"recipient_id": recipient_id, "name": overview.name})
    return _wrap_untrusted(_format_recipient_overview(overview))

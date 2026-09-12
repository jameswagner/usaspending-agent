"""The recipient_type filter's vocabulary: search_awards's recipient_type_names
filter, keyed by the actual value the live API wants.

Split out of tool_filters.py into its own file (2026-09-12) - a 70-entry data
table doesn't belong inline in a module that's otherwise logic, and this way
a future refresh against USASpending's own source is a one-file diff.

search_filters.md's own example ("Small Business", "Alaskan Native
Corporation Owned Firm" - human-readable display names) is wrong for the
current live API: live-verified 2026-09-12 that a display-name value returns
zero results against a query with known real matches, while the snake_case
key below (e.g. "small_business") returns them correctly. Keys/labels mirror
USASpending's own BUSINESS_CATEGORIES_LOOKUP_DICT (usaspending_api/common/
helpers/business_categories_helper.py); values are the human-readable label,
unused at runtime, kept only so an unrecognized-value error can show
something readable.
"""
from __future__ import annotations

from typing import Literal

from backend.app.usaspending_client import USASpendingAPIError

RECIPIENT_TYPE_NAMES: dict[str, str] = {
    "category_business": "Category Business",
    "small_business": "Small Business",
    "other_than_small_business": "Not Designated a Small Business",
    "corporate_entity_tax_exempt": "Corporate Entity Tax Exempt",
    "corporate_entity_not_tax_exempt": "Corporate Entity Not Tax Exempt",
    "partnership_or_limited_liability_partnership": "Partnership or Limited Liability Partnership",
    "sole_proprietorship": "Sole Proprietorship",
    "manufacturer_of_goods": "Manufacturer of Goods",
    "subchapter_s_corporation": "Subchapter S Corporation",
    "limited_liability_corporation": "Limited Liability Corporation",
    "minority_owned_business": "Minority Owned Business",
    "alaskan_native_corporation_owned_firm": "Alaskan Native Corporation Owned Firm",
    "american_indian_owned_business": "American Indian Owned Business",
    "asian_pacific_american_owned_business": "Asian Pacific American Owned Business",
    "black_american_owned_business": "Black American Owned Business",
    "hispanic_american_owned_business": "Hispanic American Owned Business",
    "native_american_owned_business": "Native American Owned Business",
    "native_hawaiian_organization_owned_firm": "Native Hawaiian Organization Owned Firm",
    "subcontinent_asian_indian_american_owned_business": "Indian (Subcontinent) American Owned Business",
    "tribally_owned_firm": "Tribally Owned Firm",
    "other_minority_owned_business": "Other Minority Owned Business",
    "woman_owned_business": "Woman Owned Business",
    "women_owned_small_business": "Women Owned Small Business",
    "economically_disadvantaged_women_owned_small_business": "Economically Disadvantaged Women Owned Small Business",
    "joint_venture_women_owned_small_business": "Joint Venture Women Owned Small Business",
    "joint_venture_economically_disadvantaged_women_owned_small_business":
        "Joint Venture Economically Disadvantaged Women Owned Small Business",
    "veteran_owned_business": "Veteran Owned Business",
    "service_disabled_veteran_owned_business": "Service Disabled Veteran Owned Business",
    "special_designations": "Special Designations",
    "8a_program_participant": "8(a) Program Participant",
    "ability_one_program": "AbilityOne Program Participant",
    "dot_certified_disadvantaged_business_enterprise": "DoT Certified Disadvantaged Business Enterprise",
    "emerging_small_business": "Emerging Small Business",
    "federally_funded_research_and_development_corp": "Federally Funded Research and Development Corp",
    "historically_underutilized_business_firm": "HUBZone Firm",
    "labor_surplus_area_firm": "Labor Surplus Area Firm",
    "sba_certified_8a_joint_venture": "SBA Certified 8 a Joint Venture",
    "self_certified_small_disadvanted_business": "Self-Certified Small Disadvantaged Business",
    "small_agricultural_cooperative": "Small Agricultural Cooperative",
    "small_disadvantaged_business": "Small Disadvantaged Business",
    "community_developed_corporation_owned_firm": "Community Developed Corporation Owned Firm",
    "us_owned_business": "U.S.-Owned Business",
    "foreign_owned_and_us_located_business": "Foreign-Owned and U.S.-Incorporated Business",
    "foreign_owned": "Foreign Owned",
    "foreign_government": "Foreign Government",
    "international_organization": "International Organization",
    "domestic_shelter": "Domestic Shelter",
    "hospital": "Hospital",
    "veterinary_hospital": "Veterinary Hospital",
    "nonprofit": "Nonprofit Organization",
    "foundation": "Foundation",
    "community_development_corporations": "Community Development Corporation",
    "higher_education": "Higher Education",
    "public_institution_of_higher_education": "Higher Education (Public)",
    "private_institution_of_higher_education": "Higher Education (Private)",
    "minority_serving_institution_of_higher_education": "Higher Education (Minority Serving)",
    "educational_institution": "Educational Institution",
    "school_of_forestry": "School of Forestry",
    "veterinary_college": "Veterinary College",
    "government": "Government",
    "national_government": "U.S. National Government",
    "regional_and_state_government": "U.S. Regional/State Government",
    "regional_organization": "U.S. Regional Government Organization",
    "interstate_entity": "U.S. Interstate Government Entity",
    "us_territory_or_possession": "U.S. Territory Government",
    "local_government": "U.S. Local Government",
    "indian_native_american_tribal_government": "Native American Tribal Government",
    "authorities_and_commissions": "U.S. Government Authorities",
    "council_of_governments": "Council of Governments",
    "individuals": "Individuals",
}

# Static Literal mirror of RECIPIENT_TYPE_NAMES.keys() - same reasoning as
# AwardType (tool_filters.py). TestLiteralTypesMatchVocabulary asserts this
# can't silently drift.
RecipientType = Literal[
    "category_business", "small_business", "other_than_small_business",
    "corporate_entity_tax_exempt", "corporate_entity_not_tax_exempt",
    "partnership_or_limited_liability_partnership", "sole_proprietorship",
    "manufacturer_of_goods", "subchapter_s_corporation", "limited_liability_corporation",
    "minority_owned_business", "alaskan_native_corporation_owned_firm",
    "american_indian_owned_business", "asian_pacific_american_owned_business",
    "black_american_owned_business", "hispanic_american_owned_business",
    "native_american_owned_business", "native_hawaiian_organization_owned_firm",
    "subcontinent_asian_indian_american_owned_business", "tribally_owned_firm",
    "other_minority_owned_business", "woman_owned_business", "women_owned_small_business",
    "economically_disadvantaged_women_owned_small_business",
    "joint_venture_women_owned_small_business",
    "joint_venture_economically_disadvantaged_women_owned_small_business",
    "veteran_owned_business", "service_disabled_veteran_owned_business",
    "special_designations", "8a_program_participant", "ability_one_program",
    "dot_certified_disadvantaged_business_enterprise", "emerging_small_business",
    "federally_funded_research_and_development_corp", "historically_underutilized_business_firm",
    "labor_surplus_area_firm", "sba_certified_8a_joint_venture",
    "self_certified_small_disadvanted_business", "small_agricultural_cooperative",
    "small_disadvantaged_business", "community_developed_corporation_owned_firm",
    "us_owned_business", "foreign_owned_and_us_located_business", "foreign_owned",
    "foreign_government", "international_organization", "domestic_shelter", "hospital",
    "veterinary_hospital", "nonprofit", "foundation", "community_development_corporations",
    "higher_education", "public_institution_of_higher_education",
    "private_institution_of_higher_education",
    "minority_serving_institution_of_higher_education", "educational_institution",
    "school_of_forestry", "veterinary_college", "government", "national_government",
    "regional_and_state_government", "regional_organization", "interstate_entity",
    "us_territory_or_possession", "local_government",
    "indian_native_american_tribal_government", "authorities_and_commissions",
    "council_of_governments", "individuals",
]


def _normalize_recipient_type(recipient_type: str) -> str:
    normalized = recipient_type.strip().lower().replace(" ", "_").replace("-", "_")
    if normalized not in RECIPIENT_TYPE_NAMES:
        raise USASpendingAPIError(
            f"Unknown recipient_type '{recipient_type}'. Must be one of: "
            f"{', '.join(sorted(RECIPIENT_TYPE_NAMES))}"
        )
    return normalized

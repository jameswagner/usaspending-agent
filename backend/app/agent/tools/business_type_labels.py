"""business_types code->label lookup used by get_recipient_details - pulled out of
recipients.py since the table itself dwarfs the actual logic in that file.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
    if code in BUSINESS_TYPE_LABELS:
        return BUSINESS_TYPE_LABELS[code]
    logger.warning("business_types code %r not in BUSINESS_TYPE_LABELS - falling back to title-case", code)
    return code.replace("_", " ").title()

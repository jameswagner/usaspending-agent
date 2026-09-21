"""Vocabulary for search_awards's three contract-only filters: Type of
Contract Pricing, Type of Set Aside, and Extent Competed (issue #27).
AdvancedFilters (backend/app/usaspending/filter_models.py) already models
contract_pricing_type_codes/set_aside_type_codes/extent_competed_type_codes
as raw FPDS code lists - this module is the human-key -> code lookup so the
model states "firm_fixed_price" rather than needing to know "J" is the real
FPDS code, the same normalize-then-look-up pattern as AWARD_TYPE_GROUPS/
RECIPIENT_TYPE_NAMES.

Source of the code->label pairs: the live Data Dictionary
(GET https://api.usaspending.gov/api/v2/references/data_dictionary/,
TypeOfContractPricing/ExtentCompeted/TypeSetAside rows) - the authoritative
FPDS domain values, not usaspending-website's own
src/js/dataMapping/search/contractFields.js, which was tried first and
found to have several wrong/stale code spellings for extent-competed
("E Civ", "CDOCiv", "NDOCiv" instead of the real "E", "CDO", "NDO") and
set-aside ("ISEE" instead of "IEE", plus a "Civ" suffix on "RSBCiv"/
"8ACCiv"/"HS2Civ" that doesn't exist in the real "RSB"/"8AC"/"HS2" codes) -
caught only by live-verifying every code against
/api/v2/search/spending_by_award/ rather than trusting that source as-is.

Every code below is live-verified (2026-09-19) to return real results over
the FY2008-present data history. Two set-aside codes from the Data
Dictionary's own TypeSetAside list are deliberately excluded - 8AC (SDB
Set-Aside 8(a)) and HS2 (Combination HUBZone and 8(a)) - both confirmed
live to return zero results, matching the Data Dictionary's own notes that
neither is valid for documents signed after 2005/2008 respectively.
"""
from __future__ import annotations

from typing import Literal

from backend.app.usaspending import USASpendingAPIError

CONTRACT_PRICING_TYPE_CODES: dict[str, str] = {
    "combination": "2",
    "cost_no_fee": "S",
    "cost_plus_award_fee": "R",
    "cost_plus_fixed_fee": "U",
    "cost_plus_incentive_fee": "V",
    "cost_sharing": "T",
    "firm_fixed_price": "J",
    "fixed_price_award_fee": "M",
    "fixed_price_incentive": "L",
    "fixed_price_level_of_effort": "B",
    "fixed_price_redetermination": "A",
    "fixed_price_economic_price_adjustment": "K",
    "labor_hours": "Z",
    "order_dependent": "1",
    "other": "3",
    "time_and_materials": "Y",
}

ContractPricingType = Literal[
    "combination", "cost_no_fee", "cost_plus_award_fee", "cost_plus_fixed_fee",
    "cost_plus_incentive_fee", "cost_sharing", "firm_fixed_price", "fixed_price_award_fee",
    "fixed_price_incentive", "fixed_price_level_of_effort", "fixed_price_redetermination",
    "fixed_price_economic_price_adjustment", "labor_hours", "order_dependent", "other",
    "time_and_materials",
]

# Excludes 8AC (SDB Set-Aside 8(a)) and HS2 (Combination HUBZone and 8(a)) -
# see module docstring.
SET_ASIDE_TYPE_CODES: dict[str, str] = {
    "8a_sole_source": "8AN",
    "8a_with_hubzone_preference": "HS3",
    "8a_competed": "8A",
    "buy_indian": "BI",
    "economically_disadvantaged_women_owned_small_business": "EDWOSB",
    "economically_disadvantaged_women_owned_small_business_sole_source": "EDWOSBSS",
    "emerging_small_business": "ESB",
    "hbcu_mi_partial": "HMP",
    "hbcu_mi_total": "HMT",
    "hubzone_set_aside": "HZC",
    "hubzone_sole_source": "HZS",
    "indian_economic_enterprise": "IEE",
    "indian_small_business_economic_enterprise": "ISBEE",
    "no_set_aside": "NONE",
    "reserved_for_small_business": "RSB",
    "sdvosb_sole_source": "SDVOSBS",
    "sdvosb_set_aside": "SDVOSBC",
    "small_business_set_aside_partial": "SBP",
    "small_business_set_aside_total": "SBA",
    "veteran_set_aside": "VSA",
    "veteran_sole_source": "VSS",
    "very_small_business": "VSB",
    "women_owned_small_business": "WOSB",
    "women_owned_small_business_sole_source": "WOSBSS",
}

SetAsideType = Literal[
    "8a_sole_source", "8a_with_hubzone_preference", "8a_competed", "buy_indian",
    "economically_disadvantaged_women_owned_small_business",
    "economically_disadvantaged_women_owned_small_business_sole_source",
    "emerging_small_business", "hbcu_mi_partial", "hbcu_mi_total", "hubzone_set_aside",
    "hubzone_sole_source", "indian_economic_enterprise",
    "indian_small_business_economic_enterprise", "no_set_aside", "reserved_for_small_business",
    "sdvosb_sole_source", "sdvosb_set_aside", "small_business_set_aside_partial",
    "small_business_set_aside_total", "veteran_set_aside", "veteran_sole_source",
    "very_small_business", "women_owned_small_business",
    "women_owned_small_business_sole_source",
]

EXTENT_COMPETED_TYPE_CODES: dict[str, str] = {
    "competed_under_sap": "F",
    "competitive_delivery_order": "CDO",
    "follow_on_to_competed_action": "E",
    "full_and_open_competition": "A",
    "full_and_open_competition_after_exclusion_of_sources": "D",
    "non_competitive_delivery_order": "NDO",
    "not_available_for_competition": "B",
    "not_competed": "C",
    "not_competed_under_sap": "G",
}

ExtentCompetedType = Literal[
    "competed_under_sap", "competitive_delivery_order", "follow_on_to_competed_action",
    "full_and_open_competition", "full_and_open_competition_after_exclusion_of_sources",
    "non_competitive_delivery_order", "not_available_for_competition", "not_competed",
    "not_competed_under_sap",
]


def _normalize_from_vocabulary(values: list[str], vocabulary: dict[str, str], param_name: str) -> list[str]:
    codes = []
    for value in values:
        key = value.strip().lower().replace(" ", "_").replace("-", "_")
        code = vocabulary.get(key)
        if code is None:
            raise USASpendingAPIError(
                f"Unknown {param_name} '{value}'. Must be one of: {', '.join(sorted(vocabulary))}"
            )
        codes.append(code)
    return codes


def _normalize_contract_pricing_types(values: list[str]) -> list[str]:
    return _normalize_from_vocabulary(values, CONTRACT_PRICING_TYPE_CODES, "contract_pricing_type")


def _normalize_set_aside_types(values: list[str]) -> list[str]:
    return _normalize_from_vocabulary(values, SET_ASIDE_TYPE_CODES, "set_aside_type")


def _normalize_extent_competed_types(values: list[str]) -> list[str]:
    return _normalize_from_vocabulary(values, EXTENT_COMPETED_TYPE_CODES, "extent_competed_type")

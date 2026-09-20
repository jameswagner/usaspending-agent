"""Vocabulary for search_awards's three contract-only filters: Type of
Contract Pricing, Type of Set Aside, and Extent Competed (issue #27).
AdvancedFilters (backend/app/usaspending/filter_models.py) already models
contract_pricing_type_codes/set_aside_type_codes/extent_competed_type_codes
as raw FPDS code lists - this module is the human-key -> code lookup so the
model states "firm_fixed_price" rather than needing to know "J" is the real
FPDS code, the same normalize-then-look-up pattern as AWARD_TYPE_GROUPS/
RECIPIENT_TYPE_NAMES.

Source of the code->label pairs: usaspending-website's own
src/js/dataMapping/search/contractFields.js (pricingTypeDefinitions/
setAsideDefinitions/extentCompetedDefinitions) - the same dicts that
usaspending.gov's own Advanced Search checkboxes use to build these exact
filter requests. Not every code in that source is included here: five
set-aside codes (ISEE, HS2Civ, RSBCiv, 8ACCiv, VSBCiv) and three
extent-competed codes (CDOCiv, NDOCiv, "E Civ") were live-verified
2026-09-19 against /api/v2/search/spending_by_award/ to return zero results
across the entire FY2008-present data history (every other code in each
list returned real results in the same check) - stale/legacy values not
worth exposing as a selectable filter that would always silently come back
empty.
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

# Excludes ISEE (Indian Economic Enterprise), HS2Civ (Combination HUBZone and
# 8(a)), RSBCiv (Reserved for Small Business $2,501-$100K), 8ACCiv (SDB Set
# Aside 8(a)), VSBCiv (Very Small Business Set Aside) - see module docstring.
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
    "indian_small_business_economic_enterprise": "ISBEE",
    "no_set_aside": "NONE",
    "sdvosb_sole_source": "SDVOSBS",
    "sdvosb_set_aside": "SDVOSBC",
    "small_business_set_aside_partial": "SBP",
    "small_business_set_aside_total": "SBA",
    "veteran_set_aside": "VSA",
    "veteran_sole_source": "VSS",
    "women_owned_small_business": "WOSB",
    "women_owned_small_business_sole_source": "WOSBSS",
}

SetAsideType = Literal[
    "8a_sole_source", "8a_with_hubzone_preference", "8a_competed", "buy_indian",
    "economically_disadvantaged_women_owned_small_business",
    "economically_disadvantaged_women_owned_small_business_sole_source",
    "emerging_small_business", "hbcu_mi_partial", "hbcu_mi_total", "hubzone_set_aside",
    "hubzone_sole_source", "indian_small_business_economic_enterprise", "no_set_aside",
    "sdvosb_sole_source", "sdvosb_set_aside", "small_business_set_aside_partial",
    "small_business_set_aside_total", "veteran_set_aside", "veteran_sole_source",
    "women_owned_small_business", "women_owned_small_business_sole_source",
]

# Excludes CDOCiv (Competitive Delivery Order), NDOCiv (Non-Competitive
# Delivery Order), "E Civ" (Follow On to Competed Action) - see module
# docstring.
EXTENT_COMPETED_TYPE_CODES: dict[str, str] = {
    "competed_under_sap": "F",
    "full_and_open_competition": "A",
    "full_and_open_competition_after_exclusion_of_sources": "D",
    "not_available_for_competition": "B",
    "not_competed": "C",
    "not_competed_under_sap": "G",
}

ExtentCompetedType = Literal[
    "competed_under_sap", "full_and_open_competition",
    "full_and_open_competition_after_exclusion_of_sources", "not_available_for_competition",
    "not_competed", "not_competed_under_sap",
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

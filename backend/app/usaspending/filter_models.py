"""Filter/query Pydantic models for the search endpoints' AdvancedFilterObject."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class TimePeriod(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str
    # One of action_date (default - any transaction in this window),
    # date_signed (the award's original/base transaction date), or
    # new_awards_only (only awards whose BASE transaction falls in this
    # window). Live-verified 2026-09-06: a multi-year award with activity
    # in both FY2023 and FY2024 appears in a search_awards query for
    # EITHER year under the action_date default, each time showing its
    # same cumulative Award Amount - not a bug, just what action_date
    # actually filters on. new_awards_only eliminates that recurrence
    # (verified: an award set that overlapped under action_date had zero
    # overlap under new_awards_only) at the cost of omitting ongoing
    # multi-year awards that didn't originate in the queried window.
    date_type: str | None = None


class AgencyFilter(BaseModel):
    type: Literal["awarding", "funding"]
    tier: Literal["toptier", "subtier"]
    name: str
    toptier_name: str | None = None


class AwardAmount(BaseModel):
    lower_bound: float | None = None
    upper_bound: float | None = None


class LocationObject(BaseModel):
    """StandardLocationObject per search_filters.md. Every field beyond
    country/state is modeled (cheap, and already verified from the live
    contract) even though no tool parameter exposes county/city/zip/
    district yet - modeling a field's shape and exposing it as an
    LLM-facing parameter are separate decisions; only the latter is meant
    to stay incremental/demand-driven (see ADVANCED_FILTER_FIELD_COVERAGE
    below)."""

    country: str = "USA"
    state: str | None = None
    county: str | None = None
    city: str | None = None
    district_original: str | None = None
    district_current: str | None = None
    zip: str | None = None


class NAICSCodeObject(BaseModel):
    require: list[str] | None = None
    exclude: list[str] | None = None


class CodePathObject(BaseModel):
    """Shared require/exclude-of-hierarchical-paths shape used by both
    PSCCodeObject and TASCodeObject (identical structure per
    search_filters.md, just different code vocabularies)."""

    require: list[list[str]] | None = None
    exclude: list[list[str]] | None = None


class TreasuryAccountComponentsObject(BaseModel):
    # aid/main are "required" per spending_by_category.md's version of
    # this object but optional per spending_by_award.md's
    # TASCodeComponentObject variant - the two contract docs disagree.
    # Left optional here (the more permissive reading) rather than
    # guessing which doc is authoritative; not live-verified.
    ata: str | None = None
    aid: str | None = None
    bpoa: str | None = None
    epoa: str | None = None
    a: str | None = None
    main: str | None = None
    sub: str | None = None


class ProgramActivityObject(BaseModel):
    name: str | None = None
    # code is typed as number in spending_by_award.md's ProgramActivityObject
    # but string in spending_over_time.md's - another cross-doc
    # inconsistency, not resolved by guessing; accept either.
    code: str | int | None = None


class AdvancedFilters(BaseModel):
    """Models the API's AdvancedFilterObject as completely as the live
    contracts (search_filters.md + the search/*.md endpoint contracts)
    support, as of 2026-09-06 - not just the fields this project's tools
    currently expose to the model. Modeling a field's shape is cheap and
    gives free validation to any future caller; which of these fields
    actually get an LLM-facing tool parameter is a separate, deliberately
    incremental decision (see ADVANCED_FILTER_FIELD_COVERAGE).

    extra="allow" stays as a safety net for fields the live API adds
    *after* this was last reviewed against the contracts - not as a
    substitute for modeling fields already known about.

    object_class/program_activity vs. object_classes/program_activities:
    the older spending_by_category.md/spending_over_time.md contracts and
    the newer spending_by_award.md one use different names and shapes for
    what appears to be the same underlying filter concept. Both are kept
    here rather than picking one - not live-verified which endpoints
    accept which name.
    """

    model_config = ConfigDict(extra="allow")

    keywords: list[str] | None = None
    description: str | None = None
    time_period: list[TimePeriod] | None = None
    place_of_performance_scope: Literal["domestic", "foreign"] | None = None
    place_of_performance_locations: list[LocationObject] | None = None
    agencies: list[AgencyFilter] | None = None
    recipient_search_text: list[str] | None = None
    # Not in the shared api_contracts/search_filters.md reference doc at all -
    # only in individual endpoint contracts (e.g.
    # search/spending_by_category/awarding_agency.md: "A unique identifier
    # for the recipient which includes the recipient hash and level. This
    # filter is not supported by subawards.") - found live 2026-09-08, not
    # from that shared doc (see private/HUMAN_INTERVENTIONS.md #26).
    # Confirmed live per-endpoint, not assumed universal:
    # spending_by_category and spending_over_time both honor it correctly
    # (spending_by_category: exact match to the penny against a real
    # recipient's true all-time total). search_awards/spending_by_award
    # does NOT - its own AdvancedFilterObject section doesn't list this
    # field, and live-verified it's silently ignored there (a raw curl
    # with this filter set returned completely unfiltered top-government-
    # wide contracts, with the live API's own `messages` field explicitly
    # saying so: "The following filters from the request were not used:
    # {'recipient_id'}"). Only wired into get_spending_by_category/
    # get_spending_over_time's _build_filters path - never exposed as a
    # search_awards parameter.
    recipient_id: str | None = None
    recipient_scope: Literal["domestic", "foreign"] | None = None
    recipient_locations: list[LocationObject] | None = None
    recipient_type_names: list[str] | None = None
    award_type_codes: list[str] | None = None
    award_ids: list[str] | None = None
    award_amounts: list[AwardAmount] | None = None
    program_numbers: list[str] | None = None
    naics_codes: NAICSCodeObject | None = None
    tas_codes: CodePathObject | None = None
    # psc_codes accepts either the hierarchical require/exclude object OR
    # a flat list of raw PSC code strings (search_filters.md: "Supports
    # new PSCCodeObject or legacy array of codes") - the flat form is what
    # this codebase actually uses (a single user-given code, not a tree
    # path), verified live 2026-09-06 against a real PSC-filtered query.
    psc_codes: CodePathObject | list[str] | None = None
    contract_pricing_type_codes: list[str] | None = None
    set_aside_type_codes: list[str] | None = None
    extent_competed_type_codes: list[str] | None = None
    treasury_account_components: list[TreasuryAccountComponentsObject] | None = None
    program_activities: list[ProgramActivityObject] | None = None
    object_classes: list[str] | None = None
    # Older-contract field names for the same two concepts (see docstring).
    object_class: list[str] | None = None
    program_activity: list[str] | None = None
    def_codes: list[str] | None = None
    award_unique_id: str | None = None


# Every AdvancedFilterObject field above, and whether it's actually reachable
# by a model-visible tool parameter yet. "modeled" always means "has a typed
# field on AdvancedFilters" (all of them, now) - this tracks exposure, the
# genuinely incremental/demand-driven decision, not modeling. Kept next to
# AdvancedFilters so the two can't drift apart silently; re-diffed against
# the live contracts by dev_tools/check_filter_coverage.py.
ADVANCED_FILTER_FIELD_COVERAGE: dict[str, str] = {
    "keywords": "exposed (keywords)",
    "description": "exposed (description) - distinct from keywords: phrase-prefix match against the "
                    "award's own description text only, where keywords also matches PIID/FAIN/URI and "
                    "several other text fields",
    "time_period": "exposed (start_fiscal_year/end_fiscal_year, date_type)",
    "place_of_performance_scope": "exposed (place_of_performance_scope)",
    "place_of_performance_locations": "exposed (performed_in_state)",
    "agencies": "exposed (agency_name)",
    "recipient_search_text": "exposed (recipient_name)",
    "recipient_id": "exposed (recipient_id) - get_spending_by_category/get_spending_over_time only, not "
                    "search_awards (confirmed live: silently ignored there, see this field's own comment above)",
    "recipient_scope": "exposed (recipient_scope)",
    "recipient_locations": "exposed (recipient_in_state)",
    "recipient_type_names": "exposed (recipient_type) - vocabulary is the snake_case keys from "
                             "USASpending's own BUSINESS_CATEGORIES_LOOKUP_DICT (common/helpers/"
                             "business_categories_helper.py), live-verified 2026-09-12 - NOT the "
                             "human-readable display names search_filters.md's own example shows, "
                             "which return zero results live",
    "award_type_codes": "exposed (award_type)",
    "award_ids": "exposed (award_id) - single known Award ID (PIID/FAIN/URI), fuzzy-matched, wrapped "
                 "into a 1-item list",
    "award_amounts": "exposed (min_amount/max_amount)",
    "program_numbers": "exposed (cfda_program) - direct code passthrough, not a keyword lookup: verified live that "
                        "an analyst asking about a specific CFDA program already knows the number (e.g. 10.001), the "
                        "same way NAICS/PSC codes are domain-standard identifiers, not English descriptions needing "
                        "translation the way award_type's buckets did",
    "naics_codes": "exposed (naics_code) - direct code passthrough, same reasoning as program_numbers above. A "
                    "keyword->code lookup via GET/POST /api/v2/autocomplete/naics/ exists and was verified live "
                    "(2026-09-06) to work, but wasn't built: it solves a secondary scenario (analyst doesn't know "
                    "the code) that wasn't established as the more likely one, and the endpoint's matching is a "
                    "literal substring match against official titles, not semantic (e.g. 'information technology' "
                    "and 'defense' alone both returned zero results live) - a real follow-up if demand shows up, "
                    "not a naive win to build speculatively",
    "tas_codes": "exposed (tas_codes) - direct code-path passthrough (require-list of "
                 "[ATA, AID, ...] component lists), same reasoning as naics_codes/psc_codes above",
    "psc_codes": "exposed (psc_code) - direct code passthrough, same reasoning as program_numbers/naics_codes above",
    "contract_pricing_type_codes": "modeled, not exposed",
    "set_aside_type_codes": "modeled, not exposed",
    "extent_competed_type_codes": "modeled, not exposed",
    "treasury_account_components": "exposed (federal_account) - only the aid/main pair (the federal "
                                    "account itself, e.g. '028-8704'), not the full TAS "
                                    "ata/bpoa/epoa/sub sub-components - no analyst demand observed for "
                                    "filtering by the finer-grained TAS pieces yet",
    "program_activities": "modeled, not exposed",
    "object_classes": "modeled, not exposed",
    "object_class": "modeled, not exposed (older-contract name, see AdvancedFilters docstring)",
    "program_activity": "modeled, not exposed (older-contract name, see AdvancedFilters docstring)",
    "def_codes": "exposed (def_codes) - on get_spending_by_category/get_spending_over_time/search_awards, "
                 "with client-side group-alias expansion (see tool_filters.DEFC_GROUP_ALIASES).",
    "award_unique_id": "modeled, not exposed",
}

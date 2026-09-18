"""Core Pydantic response models for the USASpending API client."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class ToptierAgency(BaseModel):
    model_config = ConfigDict(extra="allow")

    agency_id: int
    agency_name: str
    toptier_code: str
    abbreviation: str
    agency_slug: str
    active_fy: str
    active_fq: str
    budget_authority_amount: float
    obligated_amount: float
    outlay_amount: float
    percentage_of_total_budget_authority: float  # decimal (0.2356), not a percent (23.56)
    congressional_justification_url: str | None = None
    current_total_budget_authority_amount: float  # government-wide total, NOT per-agency - see AgencyYearBudget.total_budgetary_resources


class CountyMatch(BaseModel):
    model_config = ConfigDict(extra="allow")

    county_name: str
    county_fips: str  # 5-digit state+county FIPS - _normalize_county_fips strips the state prefix
    state_name: str


class LocationAutocompleteResults(BaseModel):
    model_config = ConfigDict(extra="allow")

    counties: list[CountyMatch] = []


class LocationAutocompleteResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: LocationAutocompleteResults


class AgencyAutocompleteRef(BaseModel):
    model_config = ConfigDict(extra="allow")

    # abbreviation is None live for some offices/sub-agencies without one
    # (e.g. "Bureau of Indian Affairs and Bureau of Indian Education").
    abbreviation: str | None = None
    code: str
    name: str


class SubtierAgencyMatch(AgencyAutocompleteRef):
    toptier_agency: AgencyAutocompleteRef


class AgencyOfficeAutocompleteResults(BaseModel):
    model_config = ConfigDict(extra="allow")

    toptier_agency: list[AgencyAutocompleteRef] = []
    subtier_agency: list[SubtierAgencyMatch] = []


class AgencyOfficeAutocompleteResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: AgencyOfficeAutocompleteResults


class AgencyOverview(BaseModel):
    model_config = ConfigDict(extra="allow")

    fiscal_year: int
    toptier_code: str
    name: str
    abbreviation: str | None = None
    agency_id: int
    mission: str | None = None
    website: str | None = None
    congressional_justification_url: str | None = None
    subtier_agency_count: int


class DisasterFunding(BaseModel):
    """One row of disaster/overview.md's `funding` array - budget authority under a single DEFC."""

    def_code: str
    amount: float


class DisasterSpending(BaseModel):
    """All four fields are nullable per the contract, not just optional."""

    award_obligations: float | None = None
    award_outlays: float | None = None
    total_obligations: float | None = None
    total_outlays: float | None = None


class DisasterAdditional(BaseModel):
    """Spending/budget authority not labeled with the searched DEFC but that should still count toward the total - modeled (not skipped) since it feeds get_disaster_spending_overview's own top-line total."""

    total_budget_authority: float
    spending: DisasterSpending


class DisasterOverviewResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    funding: list[DisasterFunding]
    total_budget_authority: float
    spending: DisasterSpending
    additional: DisasterAdditional | None = None


class ObligationByPeriod(BaseModel):
    period: int
    obligated: float


class AgencyYearBudget(BaseModel):
    model_config = ConfigDict(extra="allow")

    fiscal_year: int
    # The agency's own appropriated budget - what "what is X's budget"
    # actually means. Nullable per the contract.
    agency_budgetary_resources: float | None = None
    agency_total_obligated: float | None = None
    agency_total_outlayed: float | None = None
    # NOT the agency's own figure - verified live 2026-09-07: for NSF
    # FY2026, agency_budgetary_resources was ~$10.2B while this field was
    # ~$15.5 TRILLION, a >1000x gap. Per the contract: "The budget for
    # ALL agencies in the provided fiscal year" - i.e. government-wide,
    # not agency-specific. Modeled here for completeness but deliberately
    # never surfaced in get_agency_budget's formatted output - showing a
    # trillion-dollar government-wide figure next to one agency's name
    # would be exactly the kind of misleading-but-plausible number this
    # whole tool exists to prevent.
    total_budgetary_resources: float | None = None
    agency_obligation_by_period: list[ObligationByPeriod] = []


class AgencyBudgetaryResourcesResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    toptier_code: str
    agency_data_by_year: list[AgencyYearBudget]
    messages: list[str] | None = None


class SubAgencyOffice(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Contract says "required, string" but real EPA offices (e.g. code
    # 68HERH) return name=null - loosened to match reality, same pattern
    # as CategoryResult.id/GeographyTypeResult.display_name.
    name: str | None = None
    code: str
    total_obligations: float
    transaction_count: int
    new_award_count: int


class SubAgencyBreakdown(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    abbreviation: str | None = None
    total_obligations: float
    transaction_count: int
    new_award_count: int
    children: list[SubAgencyOffice] = []


class AgencySubAgencyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    toptier_code: str
    fiscal_year: int
    page_metadata: PageMetadata | None = None
    results: list[SubAgencyBreakdown]
    messages: list[str] | None = None


class CategoryResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Contract says "required, number" (the DB key), but categories with no
    # single backing DB row (e.g. naics, psc) return null in practice.
    id: int | None = None
    name: str | None = None
    code: str | None = None
    amount: float
    total_outlays: float | None = None


class PageMetadata(BaseModel):
    """Per spending_by_category.md/spending_by_award.md's PageMetadataObject.
    hasNext is the field this codebase actually cares about: whether `limit`
    silently truncated the real result set. Kept as the literal wire-format
    name (not snake_cased) rather than translated, matching this codebase's
    existing convention of keeping the API's own field names visible
    verbatim (e.g. the "Award ID"/"Recipient Name" dict keys in
    search_awards's results) rather than a renamed Python-idiomatic layer
    that would need to be remembered as a separate mapping."""

    model_config = ConfigDict(extra="allow")

    page: int
    hasNext: bool


class SpendingByCategoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str
    results: list[CategoryResult]
    limit: int
    page_metadata: PageMetadata | None = None
    messages: list[str] | None = None


class AwardTypeCounts(BaseModel):
    """spending_by_award_count.md's AwardTypeResult - six non-overlapping
    buckets covering every award (idvs kept separate from contracts here,
    unlike AWARD_TYPE_GROUPS's own "contracts" bucket which folds IDV
    sub-types in - this is the live API's own split, not ours). Confirmed
    live 2026-09-15: this is the shape for spending_level="awards" (the
    default); spending_level="subawards" returns a different subgrant/
    subcontract shape instead, not modeled here since nothing wires that
    mode through."""

    model_config = ConfigDict(extra="allow")

    contracts: int
    idvs: int
    grants: int
    direct_payments: int
    loans: int
    other: int


class SpendingByAwardCountResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: AwardTypeCounts
    spending_level: str
    messages: list[str] | None = None


class TimePeriodGroup(BaseModel):
    model_config = ConfigDict(extra="allow")

    calendar_year: str | None = None
    fiscal_year: str | None = None
    quarter: str | None = None
    month: str | None = None


class TimeResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    time_period: TimePeriodGroup
    aggregated_amount: float
    total_outlays: float | None = None


class SpendingOverTimeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    group: str
    results: list[TimeResult]
    messages: list[str] | None = None


class SearchAwardsResponse(BaseModel):
    """search_awards previously returned a bare list[dict] (data["results"]),
    silently discarding page_metadata - which meant hasNext (whether `limit`
    actually truncated a larger real result set) was thrown away before
    ever reaching the model. Found live (2026-09-06): a "contracts over
    $10M" min_amount-filtered query with the default limit came back
    hasNext=true (only 5 of the real matches returned), and the model
    presented the truncated slice as if it were the complete list - the
    same "confident but wrong/incomplete" failure shape the whole filter
    layer was built to close, just one layer further in."""

    model_config = ConfigDict(extra="allow")

    results: list[dict[str, Any]]
    page_metadata: PageMetadata | None = None
    messages: list[str] | None = None


class DEFCAmount(BaseModel):
    """A COVID/disaster-relief Disaster Emergency Fund Code breakout,
    per idvs/amounts/award_id.md. Modeled for completeness (cheap, and
    matches this codebase's convention of typing a field's shape even
    when not yet surfaced - see AdvancedFilters/ADVANCED_FILTER_FIELD_COVERAGE)
    but not read anywhere yet - same Treasury-account-level reporting
    pipeline (File C) already excluded from get_award_details's output
    elsewhere for the File C/File D reconciliation-gap reason."""

    code: str
    amount: float


class SubawardListing(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    subaward_number: str
    description: str
    action_date: str
    amount: float
    recipient_name: str


class SubawardListingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[SubawardListing]
    page_metadata: PageMetadata | None = None


class AwardFundingRow(BaseModel):
    """One row of POST /api/v2/awards/funding/ - the Federal Account
    Funding tab: which Treasury Account Symbol/object class/program
    activity combination actually paid for this award, per the File C
    (Treasury account-level) submission. Deliberately a separate call from
    get_award_details rather than merged into it - see
    _format_contract_or_idv's docstring for the File C/File D2 timing-
    and-linkage reasoning."""

    model_config = ConfigDict(extra="allow")

    federal_account: str | None = None
    account_title: str | None = None
    object_class: str | None = None
    object_class_name: str | None = None
    program_activity_code: str | int | None = None
    program_activity_name: str | None = None
    disaster_emergency_fund_code: str | None = None
    funding_agency_name: str | None = None
    awarding_agency_name: str | None = None
    transaction_obligated_amount: float | None = None
    gross_outlay_amount: float | None = None
    reporting_fiscal_year: int | None = None
    reporting_fiscal_quarter: int | None = None
    reporting_fiscal_month: int | None = None
    is_quarterly_submission: bool | None = None


class AwardFundingResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[AwardFundingRow]
    page_metadata: PageMetadata | None = None


class IDVAmountsResponse(BaseModel):
    """GET /api/v2/idvs/amounts/{award_id}/ - the actual "how much has
    been ordered under this vehicle" rollup for an IDV. An IDV's own
    total_obligation (on GET /api/v2/awards/{award_id}/) reflects only the
    vehicle's own direct activity - confirmed live 2026-09-08 that a real,
    active NSF IDIQ came back $0.00 there. The real spending sits on the
    child orders (and, for a nested vehicle, grandchild orders) placed
    against it, which is what this endpoint actually reports.

    One fixed shape (unlike GET /awards/{award_id}/, which returns one of
    three different shapes depending on award category) - modeled as a
    typed response, matching AgencyBudgetaryResourcesResponse's style,
    rather than the raw-dict pattern client.get_award uses for the
    polymorphic endpoint.

    The four *_total_account_* /*_by_defc fields are modeled for
    completeness but deliberately never surfaced in get_award_details's
    output, same reasoning as DEFCAmount above."""

    model_config = ConfigDict(extra="allow")

    generated_unique_award_id: str
    child_idv_count: int
    child_award_count: int
    child_award_total_obligation: float
    child_award_base_and_all_options_value: float
    child_award_base_exercised_options_val: float
    child_total_account_outlay: float | None = None
    child_total_account_obligation: float | None = None
    child_award_total_outlay: float | None = None
    child_account_outlays_by_defc: list[DEFCAmount] = []
    child_account_obligations_by_defc: list[DEFCAmount] = []
    grandchild_award_count: int
    grandchild_award_total_obligation: float
    grandchild_award_base_and_all_options_value: float
    grandchild_award_base_exercised_options_val: float
    grandchild_total_account_outlay: float | None = None
    grandchild_total_account_obligation: float | None = None
    grandchild_award_total_outlay: float | None = None
    grandchild_account_outlays_by_defc: list[DEFCAmount] = []
    grandchild_account_obligations_by_defc: list[DEFCAmount] = []


class RecipientListing(BaseModel):
    """One row of POST /api/v2/recipient/'s search results (recipient.md).

    amount is ALWAYS trailing-12-months - the search endpoint takes no
    `year` param at all, unlike RecipientOverview's own total_transaction_amount
    below (which respects `year`). Confirmed live 2026-09-08: Boeing's
    parent-level search-result amount ($30.3B) is dramatically smaller than
    its real all-time profile total ($439.08B) - two different time
    windows on what looks like the same kind of number, not the same
    figure at different precision. Never conflate the two when formatting.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    duns: str | None = None
    uei: str | None = None
    id: str
    amount: float
    recipient_level: Literal["R", "P", "C"]


class SearchRecipientsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    results: list[RecipientListing]
    page_metadata: PageMetadata | None = None


class RecipientLocation(BaseModel):
    model_config = ConfigDict(extra="allow")

    address_line1: str | None = None
    address_line2: str | None = None
    address_line3: str | None = None
    foreign_province: str | None = None
    city_name: str | None = None
    county_name: str | None = None
    state_code: str | None = None
    zip: str | None = None
    zip4: str | None = None
    foreign_postal_code: str | None = None
    country_name: str | None = None
    country_code: str | None = None
    congressional_code: str | None = None


class ParentRecipient(BaseModel):
    model_config = ConfigDict(extra="allow")

    parent_name: str | None = None
    parent_duns: str | None = None
    parent_id: str | None = None
    parent_uei: str | None = None


class RecipientOverview(BaseModel):
    """GET /api/v2/recipient/{recipient_id}/{?year} (recipient/recipient_id.md).

    total_transaction_amount/total_transactions and the two
    total_face_value_loan_* fields DO respect `year` (a fiscal year,
    "all", or "latest" - the trailing 12 months) - the opposite of
    RecipientListing.amount above, which never does. `name` can be the
    literal sentinel string "REDACTED DUE TO PII" - confirmed live
    2026-09-08 that this represents a shared, pooled bucket of many
    PII-redacted individual recipients, not one person (one real example:
    $14.9B, 2.24M transactions) - unlike the award side's `record_type`,
    there's no typed flag here to detect this, only the sentinel string
    itself. tools.py's formatting layer is what handles that case, not
    this model.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    alternate_names: list[str] = []
    duns: str | None = None
    uei: str | None = None
    recipient_id: str
    recipient_level: Literal["R", "P", "C"]
    parent_name: str | None = None
    parent_duns: str | None = None
    parent_id: str | None = None
    parent_uei: str | None = None
    parents: list[ParentRecipient] = []
    location: RecipientLocation | None = None
    business_types: list[str] = []
    total_transaction_amount: float
    total_transactions: int
    total_face_value_loan_amount: float
    total_face_value_loan_transactions: int


class ChildRecipient(BaseModel):
    """recipient/children/duns_or_uei.md. One row per child recipient
    under a given parent - amount respects `year` like RecipientOverview's
    total_transaction_amount, unlike RecipientListing.amount above.
    """

    model_config = ConfigDict(extra="allow")

    name: str | None = None
    duns: str | None = None
    uei: str | None = None
    recipient_id: str
    state_province: str | None = None
    amount: float


class GeographyTypeResult(BaseModel):
    """search/spending_by_geography.md. population/per_capita are computed
    by the live API itself, but reflect current-day figures regardless of
    the queried period - confirmed live 2026-09-08 (same CA population
    across FY2009/FY2016/FY2023 queries). Both are null for entries with
    no demographic data (e.g. Antarctica). display_name/shape_code are
    documented as required but confirmed live to be null for an
    "unmapped location" bucket - contract says required, reality disagrees."""

    model_config = ConfigDict(extra="allow")

    shape_code: str | None = None
    display_name: str | None = None
    aggregated_amount: float
    population: int | None = None
    per_capita: float | None = None
    total_outlays: float | None = None


class SpendingByGeographyResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    scope: str
    geo_layer: str
    spending_level: str
    results: list[GeographyTypeResult]
    messages: list[str] | None = None


class SpendingExplorerResult(BaseModel):
    """api_contracts/v2/spending.md's SpendingExplorerGeneralResponse/
    SpendingExplorerDetailedResponse, merged into one model - the two only
    differ in which optional fields are present, not their required
    shape. id/code/name are typed as required in the contract but reality
    disagrees on both counts, confirmed live: id/code come back null for
    an "Unreported Data" row (SpendingExplorerGeneralUnreportedResponse) -
    the gap between whole-of-government budgetary-resources totals and
    what's actually been reported at the requested level so far - and
    name has also been observed null for an otherwise-normal
    program_activity result (not the Unreported Data row, which does have
    a real name - "Unreported Data" itself)."""

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    code: str | None = None
    type: str
    name: str | None = None
    amount: float
    account_number: str | None = None


class SpendingExplorerResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    total: float | None = None
    end_date: str
    results: list[SpendingExplorerResult]

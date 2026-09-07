"""Typed client for the public USASpending.gov API (https://api.usaspending.gov).

Covers four endpoints, verified against the official API contracts at
https://github.com/fedspendingtransparency/usaspending-api/tree/master/usaspending_api/api_contracts/contracts/v2
(checked 2026-09-02):
  - GET  /api/v2/references/toptier_agencies/   (agency name -> code lookup)
  - GET  /api/v2/agency/{toptier_code}/         (agency overview)
  - POST /api/v2/search/spending_by_category/
  - POST /api/v2/search/spending_over_time/
  - POST /api/v2/search/spending_by_award/

`AdvancedFilters` models the filter fields most likely to be used by this
project's questions (keywords, time period, agencies, award types,
recipient text). It allows extra fields so a caller can still pass any of
the API's other filter options (naics_codes, psc_codes, tas_codes, etc.)
without every one of them being modeled here.

The public methods are @traceable - wrapping the Anthropic client
(see agent/singletons.py) already traces every tool_use/tool_result
exchange with the model, but that never sees what happens *inside* a
tool: real latency and failures of these actual network calls to
api.usaspending.gov are otherwise invisible to LangSmith entirely.
"""
from __future__ import annotations

import time
from typing import Any, Literal

import requests
from langsmith import traceable
from pydantic import BaseModel, ConfigDict
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://api.usaspending.gov"


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
    "description": "modeled, not exposed",
    "time_period": "exposed (start_fiscal_year/end_fiscal_year, date_type)",
    "place_of_performance_scope": "exposed (place_of_performance_scope)",
    "place_of_performance_locations": "exposed (performed_in_state)",
    "agencies": "exposed (agency_name)",
    "recipient_search_text": "exposed (recipient_name)",
    "recipient_scope": "exposed (recipient_scope)",
    "recipient_locations": "exposed (recipient_in_state)",
    "recipient_type_names": "modeled, not exposed - vocabulary not yet verified against a real reference list, unlike award_type/state",
    "award_type_codes": "exposed (award_type)",
    "award_ids": "modeled, not exposed",
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
    "tas_codes": "modeled, not exposed - no analyst demand observed yet",
    "psc_codes": "exposed (psc_code) - direct code passthrough, same reasoning as program_numbers/naics_codes above",
    "contract_pricing_type_codes": "modeled, not exposed",
    "set_aside_type_codes": "modeled, not exposed",
    "extent_competed_type_codes": "modeled, not exposed",
    "treasury_account_components": "modeled, not exposed",
    "program_activities": "modeled, not exposed",
    "object_classes": "modeled, not exposed",
    "object_class": "modeled, not exposed (older-contract name, see AdvancedFilters docstring)",
    "program_activity": "modeled, not exposed (older-contract name, see AdvancedFilters docstring)",
    "def_codes": "modeled, not exposed",
    "award_unique_id": "modeled, not exposed",
}


class ToptierAgency(BaseModel):
    model_config = ConfigDict(extra="allow")

    agency_id: int
    agency_name: str
    toptier_code: str
    abbreviation: str
    agency_slug: str


class AgencyOverview(BaseModel):
    model_config = ConfigDict(extra="allow")

    fiscal_year: int
    toptier_code: str
    name: str
    abbreviation: str | None = None
    agency_id: int
    mission: str | None = None
    website: str | None = None
    subtier_agency_count: int


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


class USASpendingAPIError(Exception):
    """Raised on a non-2xx response, with the API's own error detail (if any)
    as the message instead of a raw requests traceback — callers (e.g. an
    LLM tool wrapper) can surface str(e) directly without leaking internals.
    """


def _raise_with_detail(resp: requests.Response) -> None:
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        detail = None
        try:
            detail = resp.json().get("detail")
        except ValueError:
            pass
        message = detail or str(e)
        raise USASpendingAPIError(f"{resp.status_code}: {message}") from e


class USASpendingClient:
    # Toptier agencies (name -> code) change essentially never - a
    # legislative reorg, not something that happens mid-request or even
    # mid-day. 24h is a generous refresh cadence for data this static, not
    # a tuned value.
    TOPTIER_AGENCIES_CACHE_TTL_SECONDS = 24 * 60 * 60

    def __init__(self, timeout: float = 30.0):
        self.session = requests.Session()
        self.timeout = timeout
        self._toptier_agencies_cache: list[ToptierAgency] | None = None
        self._toptier_agencies_cached_at: float | None = None

        # Retry transient failures (connection errors, rate limiting, server
        # errors) with backoff. Deliberately excludes 4xx like the 404s from
        # unsupported category endpoints — those mean the request is wrong,
        # not that the server had a bad moment, so retrying is pointless.
        retry = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, path: str, params: dict | None = None) -> dict:
        resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        _raise_with_detail(resp)
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{BASE_URL}{path}", json=body, timeout=self.timeout)
        _raise_with_detail(resp)
        return resp.json()

    @traceable(run_type="tool", name="list_toptier_agencies")
    def list_toptier_agencies(self) -> list[ToptierAgency]:
        """Cached for TOPTIER_AGENCIES_CACHE_TTL_SECONDS. Found live via a
        5-agency fan-out question: find_agency_by_name calls this on every
        invocation, and this endpoint returns the same ~100-agency list
        regardless of which name is being searched for - a 5-agency
        question was re-fetching the identical, essentially-static
        reference data 5 times in one request for no reason. Cache-hit vs.
        cache-miss is now traceable directly (this method wasn't @traceable
        before - a pass-through call had nothing worth tracing, but a
        cache means there's now a real, visible difference in what
        happened between calls).

        No lock around the check-then-set: the worst case if two threads
        race is both missing the cache and both fetching the same correct
        data once each, not incorrect data - a redundant fetch this rare
        isn't worth the complexity of synchronizing it.
        """
        now = time.monotonic()
        if (
            self._toptier_agencies_cache is not None
            and self._toptier_agencies_cached_at is not None
            and now - self._toptier_agencies_cached_at < self.TOPTIER_AGENCIES_CACHE_TTL_SECONDS
        ):
            return self._toptier_agencies_cache

        data = self._get("/api/v2/references/toptier_agencies/")
        agencies = [ToptierAgency(**r) for r in data["results"]]
        self._toptier_agencies_cache = agencies
        self._toptier_agencies_cached_at = now
        return agencies

    @traceable(run_type="tool", name="find_agency_by_name")
    def find_agency_by_name(self, name: str) -> ToptierAgency | None:
        """Case-insensitive match against agency name or abbreviation.

        Tries an exact match first, then falls back to substring match, since
        callers (an LLM tool call, a user query) rarely type the full official
        agency name.
        """
        agencies = self.list_toptier_agencies()
        name_lower = name.lower()

        for a in agencies:
            if a.agency_name.lower() == name_lower or a.abbreviation.lower() == name_lower:
                return a
        for a in agencies:
            if name_lower in a.agency_name.lower():
                return a
        return None

    @traceable(run_type="tool", name="get_agency_overview")
    def get_agency_overview(self, toptier_code: str, fiscal_year: int | None = None) -> AgencyOverview:
        params = {"fiscal_year": fiscal_year} if fiscal_year else None
        data = self._get(f"/api/v2/agency/{toptier_code}/", params=params)
        return AgencyOverview(**data)

    @traceable(run_type="tool", name="spending_by_category")
    def spending_by_category(
        self,
        category: str,
        filters: AdvancedFilters,
        limit: int = 10,
        page: int = 1,
        spending_level: str = "transactions",
    ) -> SpendingByCategoryResponse:
        body = {
            "category": category,
            "filters": filters.model_dump(exclude_none=True),
            "limit": limit,
            "page": page,
            "spending_level": spending_level,
        }
        # The bare /api/v2/search/spending_by_category/ path 404s live; the
        # real endpoint is per-category, confirmed against
        # contracts/v2/search/spending_by_category/awarding_agency.md.
        data = self._post(f"/api/v2/search/spending_by_category/{category}/", body)
        return SpendingByCategoryResponse(**data)

    @traceable(run_type="tool", name="spending_over_time")
    def spending_over_time(
        self,
        filters: AdvancedFilters,
        group: str = "fiscal_year",
        subawards: bool = False,
        spending_level: str = "transactions",
    ) -> SpendingOverTimeResponse:
        body = {
            "group": group,
            "filters": filters.model_dump(exclude_none=True),
            "subawards": subawards,
            "spending_level": spending_level,
        }
        data = self._post("/api/v2/search/spending_over_time/", body)
        return SpendingOverTimeResponse(**data)

    @traceable(run_type="tool", name="search_awards_api")
    def search_awards(
        self,
        filters: AdvancedFilters,
        fields: list[str],
        limit: int = 10,
        order: str = "desc",
        sort: str | None = None,
        page: int = 1,
    ) -> SearchAwardsResponse:
        # Unlike spending_by_category/spending_over_time, award_type_codes is
        # required here per the API contract, not just optional.
        if not filters.award_type_codes:
            raise ValueError("search_awards requires filters.award_type_codes to be set")

        body: dict[str, Any] = {
            "filters": filters.model_dump(exclude_none=True),
            "fields": fields,
            "limit": limit,
            "order": order,
            "page": page,
        }
        if sort:
            body["sort"] = sort
        data = self._post("/api/v2/search/spending_by_award/", body)
        return SearchAwardsResponse(**data)

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
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import requests
from pydantic import BaseModel, ConfigDict

BASE_URL = "https://api.usaspending.gov"


class TimePeriod(BaseModel):
    start_date: str  # YYYY-MM-DD
    end_date: str


class AgencyFilter(BaseModel):
    type: Literal["awarding", "funding"]
    tier: Literal["toptier", "subtier"]
    name: str
    toptier_name: Optional[str] = None


class AdvancedFilters(BaseModel):
    """Subset of the API's AdvancedFilterObject. At least one field must be set."""

    model_config = ConfigDict(extra="allow")

    keywords: Optional[List[str]] = None
    time_period: Optional[List[TimePeriod]] = None
    agencies: Optional[List[AgencyFilter]] = None
    award_type_codes: Optional[List[str]] = None
    recipient_search_text: Optional[List[str]] = None


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
    abbreviation: Optional[str] = None
    agency_id: int
    mission: Optional[str] = None
    website: Optional[str] = None
    subtier_agency_count: int


class CategoryResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Contract says "required, number" (the DB key), but categories with no
    # single backing DB row (e.g. naics, psc) return null in practice.
    id: Optional[int] = None
    name: Optional[str] = None
    code: Optional[str] = None
    amount: float
    total_outlays: Optional[float] = None


class SpendingByCategoryResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    category: str
    results: List[CategoryResult]
    limit: int
    messages: Optional[List[str]] = None


class TimePeriodGroup(BaseModel):
    model_config = ConfigDict(extra="allow")

    calendar_year: Optional[str] = None
    fiscal_year: Optional[str] = None
    quarter: Optional[str] = None
    month: Optional[str] = None


class TimeResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    time_period: TimePeriodGroup
    aggregated_amount: float
    total_outlays: Optional[float] = None


class SpendingOverTimeResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    group: str
    results: List[TimeResult]
    messages: Optional[List[str]] = None


class USASpendingClient:
    def __init__(self, timeout: float = 30.0):
        self.session = requests.Session()
        self.timeout = timeout

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        resp = self.session.post(f"{BASE_URL}{path}", json=body, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_toptier_agencies(self) -> List[ToptierAgency]:
        data = self._get("/api/v2/references/toptier_agencies/")
        return [ToptierAgency(**r) for r in data["results"]]

    def find_agency_by_name(self, name: str) -> Optional[ToptierAgency]:
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

    def get_agency_overview(self, toptier_code: str, fiscal_year: Optional[int] = None) -> AgencyOverview:
        params = {"fiscal_year": fiscal_year} if fiscal_year else None
        data = self._get(f"/api/v2/agency/{toptier_code}/", params=params)
        return AgencyOverview(**data)

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

    def search_awards(
        self,
        filters: AdvancedFilters,
        fields: List[str],
        limit: int = 10,
        order: str = "desc",
        sort: Optional[str] = None,
        page: int = 1,
    ) -> List[Dict[str, Any]]:
        # Unlike spending_by_category/spending_over_time, award_type_codes is
        # required here per the API contract, not just optional.
        if not filters.award_type_codes:
            raise ValueError("search_awards requires filters.award_type_codes to be set")

        body: Dict[str, Any] = {
            "filters": filters.model_dump(exclude_none=True),
            "fields": fields,
            "limit": limit,
            "order": order,
            "page": page,
        }
        if sort:
            body["sort"] = sort
        data = self._post("/api/v2/search/spending_by_award/", body)
        return data["results"]

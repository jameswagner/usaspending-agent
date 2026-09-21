"""Typed client for the public USASpending.gov API (https://api.usaspending.gov).

Verified against the official API contracts at
https://github.com/fedspendingtransparency/usaspending-api/tree/master/usaspending_api/api_contracts/contracts/v2
(checked 2026-09-02, budgetary_resources added 2026-09-07):
  - GET  /api/v2/references/toptier_agencies/            (agency name -> code lookup)
  - GET  /api/v2/agency/{toptier_code}/                  (agency overview)
  - GET  /api/v2/agency/{toptier_code}/budgetary_resources/  (appropriated budget, obligations, outlays by FY)
  - POST /api/v2/search/spending_by_category/
  - POST /api/v2/search/spending_over_time/
  - POST /api/v2/search/spending_by_award/
  - POST /api/v2/search/spending_by_award_count/
  - POST /api/v2/search/spending_by_transaction/
  - POST /api/v2/autocomplete/{naics,psc,cfda}/          (verified live, not currently called by any tool)
  - POST /api/v2/autocomplete/awarding_agency_office/    (sub-tier agency resolution fallback for find_agency_by_name)

The public methods are @traceable - wrapping the Anthropic client
(see agent/singletons.py) already traces every tool_use/tool_result
exchange with the model, but that never sees what happens *inside* a
tool: real latency and failures of these actual network calls to
api.usaspending.gov are otherwise invisible to LangSmith entirely.
"""
from __future__ import annotations

import re
import time
from typing import Any

import requests
from langsmith import traceable
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .capture import _record_request
from .exceptions import _TIMEOUT_MESSAGE, USASpendingAPIError, _raise_with_detail
from .filter_models import AdvancedFilters
from .models import (
    AgencyBudgetaryResourcesResponse,
    AgencyOfficeAutocompleteResponse,
    AgencyOverview,
    AgencySubAgencyResponse,
    AwardFundingResponse,
    ChildRecipient,
    DisasterOverviewResponse,
    IDVAmountsResponse,
    LocationAutocompleteResponse,
    RecipientOverview,
    SearchAwardsResponse,
    SearchRecipientsResponse,
    SpendingByAwardCountResponse,
    SpendingByCategoryResponse,
    SpendingByGeographyResponse,
    SpendingExplorerResponse,
    SpendingOverTimeResponse,
    SubawardListingResponse,
    ToptierAgency,
    TransactionHistoryResponse,
)

BASE_URL = "https://api.usaspending.gov"


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
        try:
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=self.timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RetryError) as e:
            raise USASpendingAPIError(_TIMEOUT_MESSAGE) from e
        _record_request("GET", resp.url, None)
        _raise_with_detail(resp)
        return resp.json()

    def _post(self, path: str, body: dict) -> dict:
        try:
            resp = self.session.post(f"{BASE_URL}{path}", json=body, timeout=self.timeout)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError, requests.exceptions.RetryError) as e:
            raise USASpendingAPIError(_TIMEOUT_MESSAGE) from e
        _record_request("POST", f"{BASE_URL}{path}", body)
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

    @traceable(run_type="tool", name="autocomplete_location")
    def autocomplete_location(self, search_text: str) -> LocationAutocompleteResponse:
        """POST /api/v2/autocomplete/location/. Only counties are modeled -
        cities carry no unique code (a CityMatch is just name/state/country,
        the same shape the city filter parameter already wants directly),
        so this app has no use for the other result keys.
        """
        data = self._post("/api/v2/autocomplete/location/", {"search_text": search_text})
        return LocationAutocompleteResponse(**data)

    @traceable(run_type="tool", name="autocomplete_awarding_agency_office")
    def autocomplete_awarding_agency_office(self, search_text: str) -> AgencyOfficeAutocompleteResponse:
        """POST /api/v2/autocomplete/awarding_agency_office/. Resolves
        top-tier agencies, sub-tier agencies (NIH, CDC, IRS, ...), and
        offices in one call - unlike list_toptier_agencies, which only
        covers the ~100 cabinet-level/independent agencies. Used by
        find_agency_by_name as a fallback when a name doesn't match a
        top-tier agency directly.
        """
        data = self._post("/api/v2/autocomplete/awarding_agency_office/", {"search_text": search_text})
        return AgencyOfficeAutocompleteResponse(**data)

    @traceable(run_type="tool", name="find_agency_by_name")
    def find_agency_by_name(self, name: str) -> ToptierAgency | None:
        """Case-insensitive match against agency name or abbreviation.

        Tries an exact match first, then falls back to a whole-word substring
        match against the ~100 top-tier agencies, since callers (an LLM tool
        call, a user query) rarely type the full official agency name - e.g.
        "Education" for "Department of Education". Word boundaries matter
        here: a bare (non-boundary) substring check would match "IRS" inside
        "Department of Veterans Affairs" (the tail of "Affairs"), live-verified
        2026-09-15, which is wrong.

        On a miss, falls back to the awarding_agency_office autocomplete
        endpoint, which also covers well-known sub-tier agencies (NIH, CDC,
        FDA under HHS; IRS under Treasury; ...) - resolving to the match's
        parent top-tier agency, since every other tool (budgets, breakdowns,
        ...) keys off a toptier_code.
        """
        agencies = self.list_toptier_agencies()
        name_lower = name.lower()
        name_pattern = re.compile(rf"\b{re.escape(name_lower)}\b")

        for a in agencies:
            if a.agency_name.lower() == name_lower or a.abbreviation.lower() == name_lower:
                return a
        for a in agencies:
            if name_pattern.search(a.agency_name.lower()):
                return a

        for code in self._candidate_autocomplete_toptier_codes(name_lower):
            match = next((a for a in agencies if a.toptier_code == code), None)
            if match is not None:
                return match
        return None

    def resolve_spending_explorer_agency_id(self, agency: str) -> str | None:
        """Resolves a toptier_code or name/abbreviation to the Spending Explorer
        endpoint's own internal agency id (see spending_explorer.py's module
        docstring) - tries an exact toptier_code match first so a code can't
        spuriously hit find_agency_by_name's substring fallback.
        """
        agencies = self.list_toptier_agencies()
        exact_code = next((a for a in agencies if a.toptier_code == agency), None)
        if exact_code is not None:
            return str(exact_code.agency_id)
        match = self.find_agency_by_name(agency)
        return str(match.agency_id) if match is not None else None

    def _candidate_autocomplete_toptier_codes(self, name_lower: str):
        """Yields toptier_code candidates from the autocomplete response,
        best guess first. Two things to guard against, both live-verified
        2026-09-15:

        - The endpoint's own ranking isn't relevance-sorted for our
          purposes: searching "IRS" returns Veterans Affairs first (its
          name, "...Affairs", contains "irs" as a substring) ahead of the
          actual IRS sub-agency entry elsewhere in the same response. So
          exact abbreviation/name matches are tried before the API's own
          first result.
        - A candidate's toptier_code can be one list_toptier_agencies()
          doesn't recognize: searching "FEMA" returns it as its own
          top-tier match (code 058), but the toptier_agencies reference
          endpoint doesn't list a code-058 agency at all - FEMA only
          reports there nested under DHS (code 070), which does show up as
          a second, subtier-level match in the same response. The caller
          tries each yielded code against the real agency list in order and
          keeps going past ones that don't resolve.
        """
        results = self.autocomplete_awarding_agency_office(name_lower).results

        for t in results.toptier_agency:
            if (t.abbreviation and t.abbreviation.lower() == name_lower) or t.name.lower() == name_lower:
                yield t.code
        for s in results.subtier_agency:
            if (s.abbreviation and s.abbreviation.lower() == name_lower) or s.name.lower() == name_lower:
                yield s.toptier_agency.code

        for t in results.toptier_agency:
            yield t.code
        for s in results.subtier_agency:
            yield s.toptier_agency.code

    @traceable(run_type="tool", name="get_agency_overview")
    def get_agency_overview(self, toptier_code: str, fiscal_year: int | None = None) -> AgencyOverview:
        params = {"fiscal_year": fiscal_year} if fiscal_year else None
        data = self._get(f"/api/v2/agency/{toptier_code}/", params=params)
        return AgencyOverview(**data)

    @traceable(run_type="tool", name="get_disaster_overview")
    def get_disaster_overview(self, def_codes: list[str] | None = None) -> DisasterOverviewResponse:
        """GET /api/v2/disaster/overview/{?def_codes} - all-time totals, no fiscal_year param.
        def_codes must already be normalized (no group aliases) and is sent comma-joined,
        not as a plain list - a repeated-param list silently keeps only the last code."""
        params = {"def_codes": ",".join(def_codes)} if def_codes else None
        data = self._get("/api/v2/disaster/overview/", params=params)
        return DisasterOverviewResponse(**data)

    @traceable(run_type="tool", name="get_agency_budgetary_resources")
    def get_agency_budgetary_resources(self, toptier_code: str) -> AgencyBudgetaryResourcesResponse:
        """No fiscal_year param - the live API always returns every year it
        has (verified 2026-09-07: NSF's response covers FY2017-FY2026 in
        one call, oldest-data-availability differs from the FY2008 floor
        the other tools document - this endpoint's own history is
        shorter). Callers filter the returned list to the range they want
        in code rather than the API taking a range param, since there
        isn't one."""
        data = self._get(f"/api/v2/agency/{toptier_code}/budgetary_resources/")
        return AgencyBudgetaryResourcesResponse(**data)

    @traceable(run_type="tool", name="get_agency_sub_agency_breakdown")
    def get_agency_sub_agency_breakdown(
        self,
        toptier_code: str,
        fiscal_year: int | None = None,
        award_type_codes: list[str] | None = None,
        agency_type: str = "awarding",
        limit: int = 10,
        page: int = 1,
    ) -> AgencySubAgencyResponse:
        """Single fiscal_year, not a range - confirmed live 2026-09-09 the
        API ignores start_fiscal_year/end_fiscal_year and silently defaults
        to the current FY, unlike get_agency_budgetary_resources's own
        every-year-in-one-call shape."""
        params = {
            "fiscal_year": fiscal_year,
            "award_type_codes": award_type_codes,
            "agency_type": agency_type,
            "limit": limit,
            "page": page,
        }
        data = self._get(
            f"/api/v2/agency/{toptier_code}/sub_agency/",
            params={k: v for k, v in params.items() if v is not None},
        )
        return AgencySubAgencyResponse(**data)

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

    @traceable(run_type="tool", name="spending_by_award_count")
    def spending_by_award_count(self, filters: AdvancedFilters) -> SpendingByAwardCountResponse:
        """POST /api/v2/search/spending_by_award_count/
        (spending_by_award_count.md) - the six-way award-type count split
        (contracts/idvs/grants/direct_payments/loans/other) the real
        Advanced Search results page shows first, above any other
        breakdown. See #123.

        Confirmed live 2026-09-15: unlike spending_by_category's
        group_by="recipient", a fully unscoped whole-of-government call
        here returns fast (it's six integers, not a ranked list) - so
        this is never given a mandatory-scope requirement the way the
        other _build_filters-based tools are.
        """
        body = {"filters": filters.model_dump(exclude_none=True)}
        data = self._post("/api/v2/search/spending_by_award_count/", body)
        return SpendingByAwardCountResponse(**data)

    @traceable(run_type="tool", name="search_awards_api")
    def search_awards(
        self,
        filters: AdvancedFilters,
        fields: list[str],
        limit: int = 10,
        order: str = "desc",
        sort: str | None = None,
        page: int = 1,
        spending_level: str = "awards",
    ) -> SearchAwardsResponse:
        # Unlike spending_by_category/spending_over_time, award_type_codes is
        # required here per the API contract, not just optional.
        if not filters.award_type_codes:
            raise ValueError("search_awards requires filters.award_type_codes to be set")

        # spending_level="subawards" is confirmed live to return individual
        # subaward records (Sub-Award ID/Amount/Date/Awardee Name, Prime
        # Award ID) from this same endpoint - not a separate one. recipient_
        # search_text/recipient_locations filter the SUB-recipient in this
        # mode, not the prime - a real, opposite-of-normal semantic, see
        # search_subawards's own docstring. recipient_id is confirmed live
        # to be silently ignored for subawards (the API's own `messages`
        # field says so), same as it already is for prime award search.
        body: dict[str, Any] = {
            "filters": filters.model_dump(exclude_none=True),
            "fields": fields,
            "limit": limit,
            "order": order,
            "page": page,
            "spending_level": spending_level,
        }
        if sort:
            body["sort"] = sort
        data = self._post("/api/v2/search/spending_by_award/", body)
        return SearchAwardsResponse(**data)

    @traceable(run_type="tool", name="search_transactions_api")
    def search_transactions(
        self,
        filters: AdvancedFilters,
        fields: list[str],
        limit: int = 10,
        order: str = "desc",
        sort: str | None = None,
        page: int = 1,
    ) -> SearchAwardsResponse:
        """POST /api/v2/search/spending_by_transaction/ - the transaction-
        level counterpart to search_awards: one row per transaction/
        modification (Award ID, Mod, Recipient Name, Action Date,
        Transaction Amount, ...) rather than one row per award with a
        cumulative lifetime total. This is what USASpending.gov's own
        "Keyword Search" results page shows, distinct from Advanced
        Search's award-rollup shape search_awards returns (issue #23).

        Reuses SearchAwardsResponse - live-verified 2026-09-19 the response
        shape (results: list[dict], page_metadata, messages) is identical
        to search_awards's own. Same required-award_type_codes contract as
        search_awards.
        """
        if not filters.award_type_codes:
            raise ValueError("search_transactions requires filters.award_type_codes to be set")

        body: dict[str, Any] = {
            "filters": filters.model_dump(exclude_none=True),
            "fields": fields,
            "limit": limit,
            "order": order,
            "page": page,
        }
        if sort:
            body["sort"] = sort
        data = self._post("/api/v2/search/spending_by_transaction/", body)
        return SearchAwardsResponse(**data)

    @traceable(run_type="tool", name="get_award")
    def get_award(self, award_id: str) -> dict[str, Any]:
        """Raw passthrough of GET /api/v2/awards/{award_id}/ - not modeled
        as a single Pydantic type like the other _raw functions' responses,
        since the actual shape returned (ContractResponse, IDVResponse, or
        FinancialAssistanceResponse per award_id.md) varies by the award's
        category and this app only surfaces a curated subset of each; the
        formatting layer (tools.py's _format_award_details) picks out and
        interprets the fields it needs directly from this dict.

        award_id must be the hash-style generated_unique_award_id (e.g.
        "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-"), not the plain PIID/
        FAIN search_awards's own "Award ID" field shows - confirmed live
        2026-09-08 that the plain PIID/FAIN 404s here.
        """
        return self._get(f"/api/v2/awards/{award_id}/")

    @traceable(run_type="tool", name="get_idv_amounts")
    def get_idv_amounts(self, award_id: str) -> IDVAmountsResponse:
        """The child/grandchild-order rollup for one IDV - see
        IDVAmountsResponse's docstring for why this is a separate call
        from get_award rather than something get_award itself returns.
        Same award_id format as get_award (hash-style
        generated_unique_award_id, not the plain PIID)."""
        data = self._get(f"/api/v2/idvs/amounts/{award_id}/")
        return IDVAmountsResponse(**data)

    @traceable(run_type="tool", name="get_award_subawards")
    def get_award_subawards(
        self, award_id: str, limit: int = 10, page: int = 1, sort: str = "amount", order: str = "desc"
    ) -> SubawardListingResponse:
        """POST /api/v2/subawards/ - subawards under one specific prime
        award, for the award-profile page's own Sub-Awards tab. Same
        award_id format as get_award (the hash-style
        generated_unique_award_id, not the plain PIID/FAIN) - a different
        endpoint from search_awards's spending_level="subawards" mode,
        which searches across all subawards rather than listing one
        award's own.
        """
        body = {"award_id": award_id, "limit": limit, "page": page, "sort": sort, "order": order}
        data = self._post("/api/v2/subawards/", body)
        return SubawardListingResponse(**data)

    @traceable(run_type="tool", name="get_award_funding")
    def get_award_funding(
        self,
        award_id: str,
        limit: int = 10,
        page: int = 1,
        sort: str = "reporting_fiscal_date",
        order: str = "desc",
    ) -> AwardFundingResponse:
        """POST /api/v2/awards/funding/ - the Federal Account Funding tab
        for one specific award: which Treasury Account
        Symbol/object class/program activity/DEFC combinations actually
        funded it, per row. Same award_id format as get_award (the
        hash-style generated_unique_award_id, not the plain PIID/FAIN).
        A separately-timed File C submission from the award's own File D2
        total_obligation (get_award) - not guaranteed to reconcile to the
        penny, per usaspending-api's own C_to_D_Linkage.md."""
        body = {"award_id": award_id, "limit": limit, "page": page, "sort": sort, "order": order}
        data = self._post("/api/v2/awards/funding/", body)
        return AwardFundingResponse(**data)

    @traceable(run_type="tool", name="get_award_transaction_history")
    def get_award_transaction_history(
        self, award_id: str, limit: int = 10, page: int = 1, sort: str = "action_date", order: str = "desc"
    ) -> TransactionHistoryResponse:
        """POST /api/v2/transactions/ - the award-profile page's own
        Transaction History tab: every individual modification/transaction
        that built up to this award's current state (mod number, action
        date, action type, amount, description). Same award_id format as
        get_award (the hash-style generated_unique_award_id, not the plain
        PIID/FAIN) - live-verified 2026-09-18 that a plain PIID doesn't 404
        here like get_award does, it silently returns an empty results
        list, so an empty response isn't on its own proof the award_id is
        wrong. live-verified sort accepts more values than transactions.md
        documents (also: id, type, type_description, action_type, is_fpds,
        cfda_number), but this app only exposes the contract-documented
        default (action_date)."""
        body = {"award_id": award_id, "limit": limit, "page": page, "sort": sort, "order": order}
        data = self._post("/api/v2/transactions/", body)
        return TransactionHistoryResponse(**data)

    @traceable(run_type="tool", name="search_recipients")
    def search_recipients(
        self,
        keyword: str | None = None,
        award_type: str = "all",
        limit: int = 10,
        sort: str = "amount",
        order: str = "desc",
        page: int = 1,
    ) -> SearchRecipientsResponse:
        """POST /api/v2/recipient/ - keyword search over recipient name/UEI/
        DUNS (recipient.md). Deliberately not a hidden resolution helper
        like find_agency_by_name - live-verified 2026-09-08 that a plain
        company name is genuinely ambiguous at this scale (546 total
        matches for "Leidos", 6+ distinct recipient_ids sharing the
        identical display name "LEIDOS, INC." alone), so the caller needs
        to see every candidate, not have one silently picked.

        award_type here is a real, different, coarser vocabulary than
        AWARD_TYPE_GROUPS (6 values: all/contracts/grants/loans/
        direct_payments/other_financial_assistance - no sub-type
        granularity) - see RecipientAwardType in tool_filters.py.

        keyword=None omits the field (the API rejects "" but accepts it being absent,
        returning an unscoped, globally-ranked list).
        """
        body: dict[str, Any] = {"award_type": award_type, "limit": limit, "sort": sort, "order": order, "page": page}
        if keyword is not None:
            body["keyword"] = keyword
        data = self._post("/api/v2/recipient/", body)
        return SearchRecipientsResponse(**data)

    @traceable(run_type="tool", name="get_recipient")
    def get_recipient(self, recipient_id: str, year: str | None = None) -> RecipientOverview:
        """GET /api/v2/recipient/{recipient_id}/{?year} (recipient/recipient_id.md).
        year is a fiscal year, "all", or "latest" (trailing 12 months).
        Confirmed live 2026-09-08: omitting year entirely gives a total
        matching year="latest" (within a fraction of a percent - the small
        remaining gap is just the trailing-12-months window itself moving
        between two separate real-time calls, not a different scope) - NOT
        year="all", which is dramatically larger ($30.3B vs $439.08B for
        the same Boeing recipient_id). Since that default duplicates what
        RecipientListing.amount from search_recipients already shows,
        get_recipient_details in tools.py defaults its own year param to
        "all" rather than leaving it unset, so the profile call gives a
        genuinely different, additive answer."""
        params = {"year": year} if year else None
        data = self._get(f"/api/v2/recipient/{recipient_id}/", params=params)
        return RecipientOverview(**data)

    @traceable(run_type="tool", name="get_recipient_children")
    def get_recipient_children(self, duns_or_uei: str, year: str | None = None) -> list[ChildRecipient]:
        """GET /api/v2/recipient/children/{duns_or_uei}/{?year}
        (recipient/children/duns_or_uei.md). Keyed by DUNS or UEI, not
        recipient_id - a third identifier space from RecipientOverview's
        own recipient_id. Confirmed live 2026-09-15: the response is a
        bare, unsorted array (Boeing's 123 children not in amount order),
        with no sort/order params and no pagination - not present in the
        upstream contract, and passing them live had no effect."""
        params = {"year": year} if year else None
        data = self._get(f"/api/v2/recipient/children/{duns_or_uei}/", params=params)
        return [ChildRecipient(**c) for c in data]

    @traceable(run_type="tool", name="spending_by_geography")
    def spending_by_geography(
        self,
        filters: AdvancedFilters,
        scope: str,
        geo_layer: str,
        geo_layer_filters: list[str] | None = None,
    ) -> SpendingByGeographyResponse:
        """POST /api/v2/search/spending_by_geography/. spending_level is
        deliberately not a parameter here - hardcoded to "transactions",
        the only variant confirmed live to be additive across separate
        period-scoped calls ("awards" mode overcounted by 60%+ in testing
        unless paired with date_type="new_awards_only", which isn't worth
        the complexity of exposing)."""
        body: dict[str, Any] = {
            "filters": filters.model_dump(exclude_none=True),
            "scope": scope,
            "geo_layer": geo_layer,
            "spending_level": "transactions",
        }
        if geo_layer_filters:
            body["geo_layer_filters"] = geo_layer_filters
        data = self._post("/api/v2/search/spending_by_geography/", body)
        return SpendingByGeographyResponse(**data)

    @traceable(run_type="tool", name="spending_explorer")
    def spending_explorer(self, explorer_type: str, filters: dict[str, str]) -> SpendingExplorerResponse:
        """POST /api/v2/spending/ - a genuinely different data lineage
        (whole-of-government File A/B account data) from every other
        method on this client (File C/D award-search data), per
        api_contracts/v2/spending.md. filters is a plain dict, not an
        AdvancedFilters - this endpoint's filter shape (fy/quarter plus
        the drill-down ids) doesn't overlap with the award-search filter
        object at all."""
        body = {"type": explorer_type, "filters": filters}
        data = self._post("/api/v2/spending/", body)
        return SpendingExplorerResponse(**data)

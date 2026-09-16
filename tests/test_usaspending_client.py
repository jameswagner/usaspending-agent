import json
from typing import Any, ClassVar

import pytest
import requests

from backend.app.usaspending_client import (
    AdvancedFilters,
    AgencySubAgencyResponse,
    ChildRecipient,
    DisasterOverviewResponse,
    RecipientOverview,
    SpendingByAwardCountResponse,
    ToptierAgency,
    USASpendingAPIError,
    USASpendingClient,
    _raise_with_detail,
    drain_request_capture,
)


def make_response(status_code: int, json_body=None, reason: str = "Error") -> requests.Response:
    resp = requests.Response()
    resp.status_code = status_code
    resp.reason = reason
    resp._content = json.dumps(json_body).encode("utf-8") if json_body is not None else b"not json"
    return resp


class TestRaiseWithDetail:
    def test_2xx_does_not_raise(self):
        _raise_with_detail(make_response(200, {"ok": True}))  # should not raise

    def test_error_with_json_detail_uses_that_message(self):
        # this is the real shape the live API returns, e.g. the FY1999
        # date-range validation error found during testing
        body = {"detail": "start_date falls before the earliest available search date of 2007-10-01."}
        with pytest.raises(USASpendingAPIError) as exc_info:
            _raise_with_detail(make_response(422, body))
        assert "start_date falls before the earliest available search date" in str(exc_info.value)

    def test_error_without_json_body_falls_back_to_requests_message(self):
        resp = make_response(404, json_body=None)
        with pytest.raises(USASpendingAPIError) as exc_info:
            _raise_with_detail(resp)
        assert "404" in str(exc_info.value)

    def test_error_with_json_body_but_no_detail_key_falls_back(self):
        with pytest.raises(USASpendingAPIError) as exc_info:
            _raise_with_detail(make_response(500, {"something_else": "value"}))
        assert "500" in str(exc_info.value)


def make_agency(name: str, abbreviation: str, code: str = "000") -> ToptierAgency:
    return ToptierAgency(
        agency_id=1,
        agency_name=name,
        toptier_code=code,
        abbreviation=abbreviation,
        agency_slug=name.lower().replace(" ", "-"),
        active_fy="2026",
        active_fq="4",
        budget_authority_amount=0.0,
        obligated_amount=0.0,
        outlay_amount=0.0,
        percentage_of_total_budget_authority=0.0,
        current_total_budget_authority_amount=0.0,
    )


class TestFindAgencyByName:
    @pytest.fixture
    def client(self):
        return USASpendingClient()

    @pytest.fixture
    def agencies(self):
        return [
            make_agency("National Science Foundation", "NSF", code="049"),
            make_agency("Department of Education", "ED", code="091"),
            make_agency("National Aeronautics and Space Administration", "NASA", code="080"),
        ]

    def test_exact_name_match(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        result = client.find_agency_by_name("National Science Foundation")
        assert result.toptier_code == "049"

    def test_exact_name_match_is_case_insensitive(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        result = client.find_agency_by_name("national science foundation")
        assert result.toptier_code == "049"

    def test_exact_abbreviation_match(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        result = client.find_agency_by_name("NASA")
        assert result.toptier_code == "080"

    def test_substring_match_fallback(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        result = client.find_agency_by_name("Education")
        assert result.toptier_code == "091"

    def test_no_match_falls_back_to_autocomplete_and_returns_none_on_a_miss(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        empty = {"results": {"toptier_agency": [], "subtier_agency": [], "office": []}, "messages": []}
        monkeypatch.setattr(client, "_post", lambda path, body: empty)
        result = client.find_agency_by_name("Department of Pizza")
        assert result is None

    def test_exact_match_wins_over_substring_match_on_a_different_agency(self, client, monkeypatch):
        # "NSF" is an exact abbreviation match on one agency and also a
        # substring of a hypothetical differently-named agency; exact match
        # must win regardless of list order
        agencies = [
            make_agency("Something NSF Adjacent Council", "SNAC", code="111"),
            make_agency("National Science Foundation", "NSF", code="049"),
        ]
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        result = client.find_agency_by_name("NSF")
        assert result.toptier_code == "049"

    def test_subtier_agency_name_resolves_to_parent_toptier_agency(self, client, monkeypatch):
        # Real shape (verified live 2026-09-15): NIH has no entry of its own
        # in list_toptier_agencies, only HHS does - the fallback must resolve
        # NIH's toptier_agency.code ("075") back to the already-fetched HHS
        # ToptierAgency object.
        hhs = make_agency("Department of Health and Human Services", "HHS", code="075")
        agencies = [hhs, make_agency("National Science Foundation", "NSF", code="049")]
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        body = {
            "results": {
                "toptier_agency": [],
                "subtier_agency": [
                    {
                        "abbreviation": "NIH",
                        "code": "7529",
                        "name": "National Institutes of Health",
                        "offices": [],
                        "toptier_agency": {"abbreviation": "HHS", "code": "075", "name": "Department of Health and Human Services"},
                    }
                ],
                "office": [],
            },
            "messages": [],
        }
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        result = client.find_agency_by_name("NIH")
        assert result is hhs

    def test_autocomplete_toptier_match_resolves_directly(self, client, monkeypatch):
        # A search term that only the autocomplete endpoint's fuzzier
        # matching turns up (missed by the plain substring pass above) can
        # also land directly in the toptier_agency bucket, not just
        # subtier_agency.
        hhs = make_agency("Department of Health and Human Services", "HHS", code="075")
        agencies = [hhs]
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
        body = {
            "results": {
                "toptier_agency": [
                    {
                        "abbreviation": "HHS",
                        "code": "075",
                        "name": "Department of Health and Human Services",
                        "subtier_agencies": [],
                    }
                ],
                "subtier_agency": [],
                "office": [],
            },
            "messages": [],
        }
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        # deliberately not an exact or substring match against the fixture
        # agency, so this only resolves via the autocomplete fallback
        result = client.find_agency_by_name("Health & Human Svcs")
        assert result is hhs


class TestListToptierAgenciesCaching:
    # Found live: a 5-agency fan-out question re-fetched this same,
    # essentially-static ~100-agency list 5 times in one request, since
    # find_agency_by_name calls this on every invocation with no caching.

    @pytest.fixture
    def client(self):
        return USASpendingClient()

    def _fake_get_counting(self, call_count: dict):
        def fake_get(path, params=None):
            call_count["n"] += 1
            return {
                "results": [
                    {
                        "agency_id": 1,
                        "agency_name": "National Science Foundation",
                        "toptier_code": "049",
                        "abbreviation": "NSF",
                        "agency_slug": "nsf",
                        "active_fy": "2026",
                        "active_fq": "4",
                        "budget_authority_amount": 0.0,
                        "obligated_amount": 0.0,
                        "outlay_amount": 0.0,
                        "percentage_of_total_budget_authority": 0.0,
                        "current_total_budget_authority_amount": 0.0,
                    }
                ]
            }

        return fake_get

    def test_second_call_within_ttl_does_not_refetch(self, client, monkeypatch):
        call_count = {"n": 0}
        monkeypatch.setattr(client, "_get", self._fake_get_counting(call_count))

        client.list_toptier_agencies()
        client.list_toptier_agencies()

        assert call_count["n"] == 1

    def test_refetches_after_ttl_expires(self, client, monkeypatch):
        call_count = {"n": 0}
        monkeypatch.setattr(client, "_get", self._fake_get_counting(call_count))

        client.list_toptier_agencies()
        # Simulate the TTL elapsing by backdating the cache timestamp,
        # rather than actually sleeping in a test.
        client._toptier_agencies_cached_at -= USASpendingClient.TOPTIER_AGENCIES_CACHE_TTL_SECONDS + 1
        client.list_toptier_agencies()

        assert call_count["n"] == 2

    def test_returns_parsed_agencies(self, client, monkeypatch):
        call_count = {"n": 0}
        monkeypatch.setattr(client, "_get", self._fake_get_counting(call_count))

        result = client.list_toptier_agencies()

        assert len(result) == 1
        assert result[0].agency_name == "National Science Foundation"

    def test_parses_real_response_shape(self, client, monkeypatch):
        # Real live values (HHS, 2026-09-09) - confirmed current_total_budget_authority_amount
        # is identical across every agency (a government-wide total, not HHS's own figure),
        # unlike budget_authority_amount/percentage_of_total_budget_authority which are real
        # per-agency figures matching the live Agency Profile page's own displayed numbers.
        body = {
            "results": [
                {
                    "agency_id": 168, "agency_name": "Department of Health and Human Services (HHS)",
                    "toptier_code": "075", "abbreviation": "HHS", "agency_slug": "health-and-human-services",
                    "active_fy": "2026", "active_fq": "4",
                    "budget_authority_amount": 3650342489549.92,
                    "obligated_amount": 0.0, "outlay_amount": 0.0,
                    "percentage_of_total_budget_authority": 0.2355772266133647,
                    "congressional_justification_url": "https://www.hhs.gov/cj",
                    "current_total_budget_authority_amount": 15495311418794.12,
                },
            ]
        }
        monkeypatch.setattr(client, "_get", lambda path, params=None: body)
        result = client.list_toptier_agencies()
        assert result[0].budget_authority_amount == 3650342489549.92
        assert result[0].percentage_of_total_budget_authority == pytest.approx(0.2355772266133647)
        assert result[0].congressional_justification_url == "https://www.hhs.gov/cj"


class TestAutocompleteLocation:
    def test_parses_real_response_shape(self, monkeypatch):
        # Real live values (Yavapai County, AZ, 2026-09-10).
        client = USASpendingClient()
        body = {
            "count": 1,
            "results": {
                "cities": [
                    {"city_name": "YAVAPAI HILLS", "state_name": "ARIZONA", "country_name": "UNITED STATES"},
                ],
                "counties": [
                    {"county_name": "YAVAPAI", "county_fips": "04025", "state_name": "ARIZONA", "country_name": "UNITED STATES"},
                ],
            },
            "messages": [""],
        }
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        response = client.autocomplete_location("Yavapai")
        assert response.results.counties[0].county_name == "YAVAPAI"
        assert response.results.counties[0].county_fips == "04025"
        assert response.results.counties[0].state_name == "ARIZONA"

    def test_no_county_matches_parses_to_empty_list(self, monkeypatch):
        # Real live shape for a query with only city matches, e.g. "Portland".
        client = USASpendingClient()
        body = {"count": 1, "results": {"cities": [{"city_name": "PORTLAND", "country_name": "CANADA"}]}, "messages": [""]}
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        response = client.autocomplete_location("Portland")
        assert response.results.counties == []


class TestSearchAwardsValidation:
    def test_raises_without_award_type_codes(self):
        client = USASpendingClient()
        with pytest.raises(ValueError, match="award_type_codes"):
            client.search_awards(AdvancedFilters(keywords=["test"]), fields=["Award ID"])

    def test_spending_level_defaults_to_awards(self, monkeypatch):
        client = USASpendingClient()
        captured = {}
        monkeypatch.setattr(client, "_post", lambda path, body: (captured.update(body), {"results": []})[1])
        client.search_awards(AdvancedFilters(award_type_codes=["A"]), fields=["Award ID"])
        assert captured["spending_level"] == "awards"

    def test_spending_level_subawards_is_passed_through(self, monkeypatch):
        # Confirmed live: the same search/spending_by_award/ endpoint
        # returns individual subaward records when this is set - not a
        # separate endpoint.
        client = USASpendingClient()
        captured = {}
        monkeypatch.setattr(client, "_post", lambda path, body: (captured.update(body), {"results": []})[1])
        client.search_awards(AdvancedFilters(award_type_codes=["A"]), fields=["Sub-Award ID"], spending_level="subawards")
        assert captured["spending_level"] == "subawards"


class TestGetAwardSubawards:
    def test_parses_real_response_shape(self, monkeypatch):
        # Real live values (Electric Boat SSN 792 contract, 2026-09-10).
        client = USASpendingClient()
        body = {
            "page_metadata": {"page": 1, "next": 2, "previous": None, "hasNext": True, "hasPrevious": False},
            "results": [
                {
                    "id": 12345,
                    "subaward_number": "PPD090=051",
                    "description": "BATTERY KIT",
                    "action_date": "2017-05-08",
                    "amount": 12652399.44,
                    "recipient_name": "STRYTEN ENERGY LLC",
                },
            ],
        }
        captured = {}
        monkeypatch.setattr(client, "_post", lambda path, b: (captured.update({"path": path, "body": b}), body)[1])
        response = client.get_award_subawards("CONT_AWD_N0002412C2115_9700_-NONE-_-NONE-", limit=3)
        assert captured["path"] == "/api/v2/subawards/"
        assert captured["body"]["award_id"] == "CONT_AWD_N0002412C2115_9700_-NONE-_-NONE-"
        assert response.page_metadata.hasNext is True
        assert response.results[0].recipient_name == "STRYTEN ENERGY LLC"
        assert response.results[0].amount == 12652399.44


class TestRecipientClientMethods:
    # Real live response shapes (Boeing, 2026-09-08), not synthetic -
    # confirms the client parses the actual wire format, not just
    # whatever a hand-written fixture happens to match.

    def test_search_recipients_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        body = {
            "page_metadata": {"page": 1, "total": 546, "limit": 10, "hasNext": True, "hasPrevious": False},
            "results": [
                {
                    "id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P",
                    "duns": "009256819", "uei": "NU2UC8MX6NK1",
                    "name": "THE BOEING COMPANY", "recipient_level": "P", "amount": 30309729588.71,
                },
            ],
        }
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        response = client.search_recipients("Boeing")
        assert response.page_metadata.hasNext is True
        assert response.results[0].name == "THE BOEING COMPANY"
        assert response.results[0].recipient_level == "P"

    def test_search_recipients_omits_keyword_when_none(self, monkeypatch):
        client = USASpendingClient()
        captured: dict = {}

        def fake_post(path, body):
            captured.update(body)
            return {"page_metadata": {"page": 1, "total": 0, "limit": 10, "hasNext": False, "hasPrevious": False}, "results": []}

        monkeypatch.setattr(client, "_post", fake_post)
        client.search_recipients()
        assert "keyword" not in captured

    def test_get_recipient_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        body = {
            "name": "REDACTED DUE TO PII", "alternate_names": [], "duns": None, "uei": None,
            "recipient_id": "6e4362a8-7dd7-8d86-d2ff-8faa5eefe0aa-R", "recipient_level": "R",
            "parent_name": None, "parent_duns": None, "parent_id": None, "parent_uei": None, "parents": [],
            "location": {"city_name": "PLANT CITY", "state_code": "FL", "country_name": "UNITED STATES"},
            "business_types": [],
            "total_transaction_amount": 14894373724.28, "total_transactions": 2243854,
            "total_face_value_loan_amount": 12841356171.82, "total_face_value_loan_transactions": 76040,
        }
        monkeypatch.setattr(client, "_get", lambda path, params=None: body)
        overview = client.get_recipient("6e4362a8-7dd7-8d86-d2ff-8faa5eefe0aa-R")
        assert isinstance(overview, RecipientOverview)
        assert overview.name == "REDACTED DUE TO PII"
        assert overview.total_transactions == 2243854

    def test_get_recipient_children_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        body = [
            {
                "recipient_id": "241d908a-e5e9-f275-b625-6b1f76859002-C",
                "name": "THE BOEING COMPANY", "duns": "832963495", "uei": "MF2LE5RK6L84",
                "amount": 23975032530.57, "state_province": "MO",
            },
            {
                "recipient_id": "e7360887-4da9-617b-357b-1c3446728140-C",
                "name": "BOEING DISTRIBUTION SERVICES X, INC.", "duns": "069172294", "uei": "X3K6MA9ZLTW6",
                "amount": 62785867.7, "state_province": "FL",
            },
        ]
        monkeypatch.setattr(client, "_get", lambda path, params=None: body)
        children = client.get_recipient_children("NU2UC8MX6NK1", year="all")
        assert len(children) == 2
        assert all(isinstance(c, ChildRecipient) for c in children)
        assert children[0].name == "THE BOEING COMPANY"
        assert children[1].amount == 62785867.7


class TestSpendingByAwardCount:
    # Real live response shape (NSF FY2024, 2026-09-15), not synthetic -
    # see #123.

    def test_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        body = {
            "results": {"contracts": 810, "direct_payments": 0, "grants": 28420, "idvs": 37, "loans": 0, "other": 0},
            "spending_level": "awards",
            "messages": ["For searches, time period start and end dates are currently limited..."],
        }
        monkeypatch.setattr(client, "_post", lambda path, b: body)
        response = client.spending_by_award_count(AdvancedFilters())
        assert isinstance(response, SpendingByAwardCountResponse)
        assert response.results.contracts == 810
        assert response.results.grants == 28420
        assert response.results.idvs == 37
        assert response.messages

    def test_posts_to_award_count_endpoint(self, monkeypatch):
        client = USASpendingClient()
        captured: dict = {}

        def fake_post(path, body):
            captured["path"] = path
            captured["body"] = body
            return {
                "results": {"contracts": 0, "direct_payments": 0, "grants": 0, "idvs": 0, "loans": 0, "other": 0},
                "spending_level": "awards",
            }

        monkeypatch.setattr(client, "_post", fake_post)
        client.spending_by_award_count(AdvancedFilters())
        assert captured["path"] == "/api/v2/search/spending_by_award_count/"
        assert "filters" in captured["body"]


class TestGetDisasterOverview:
    # Real live response shape (def_codes=L, 2026-09-15), not synthetic - see #110.

    def test_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        body = {
            "funding": [{"def_code": "L", "amount": 7707863149.53}],
            "total_budget_authority": 7707863149.53,
            "spending": {
                "award_obligations": 4333842060.96,
                "award_outlays": 6249313278.94,
                "total_obligations": 7406214022.0,
                "total_outlays": 7073955847.63,
            },
            "additional": None,
        }
        monkeypatch.setattr(client, "_get", lambda path, params=None: body)
        response = client.get_disaster_overview(["L"])
        assert isinstance(response, DisasterOverviewResponse)
        assert response.funding[0].def_code == "L"
        assert response.total_budget_authority == 7707863149.53
        assert response.spending.award_obligations == 4333842060.96
        assert response.additional is None

    def test_parses_additional_block_when_present(self, monkeypatch):
        # Per disaster/overview.md's own example - live-verified 2026-09-15 that
        # a plain def_codes=L query returns additional=None, but the field is
        # real (not speculative) and needs to parse correctly when it does show up.
        client = USASpendingClient()
        body = {
            "funding": [{"def_code": "Z", "amount": 11230000000}],
            "total_budget_authority": 11230000000,
            "spending": {
                "award_obligations": 866700000000,
                "award_outlays": 413100000000,
                "total_obligations": 963000000000,
                "total_outlays": 459000000000,
            },
            "additional": {
                "total_budget_authority": 789000000,
                "spending": {"total_obligations": 45600000, "total_outlays": 12300000},
            },
        }
        monkeypatch.setattr(client, "_get", lambda path, params=None: body)
        response = client.get_disaster_overview(["Z"])
        assert response.additional is not None
        assert response.additional.total_budget_authority == 789000000
        assert response.additional.spending.total_obligations == 45600000

    def test_sends_comma_joined_def_codes_not_a_list(self, monkeypatch):
        # Confirmed live: requests' default list encoding (repeated ?def_codes=L&
        # def_codes=M query params) silently returns results for only the LAST
        # code - a real undercount bug, not just a style preference. See
        # get_disaster_overview's own docstring.
        client = USASpendingClient()
        captured: dict = {}

        def fake_get(path, params=None):
            captured["path"] = path
            captured["params"] = params
            return {
                "funding": [],
                "total_budget_authority": 0.0,
                "spending": {
                    "award_obligations": 0.0,
                    "award_outlays": 0.0,
                    "total_obligations": 0.0,
                    "total_outlays": 0.0,
                },
                "additional": None,
            }

        monkeypatch.setattr(client, "_get", fake_get)
        client.get_disaster_overview(["L", "M"])
        assert captured["path"] == "/api/v2/disaster/overview/"
        assert captured["params"] == {"def_codes": "L,M"}

    def test_no_def_codes_sends_no_params(self, monkeypatch):
        client = USASpendingClient()
        captured: dict = {}

        def fake_get(path, params=None):
            captured["params"] = params
            return {
                "funding": [],
                "total_budget_authority": 0.0,
                "spending": {
                    "award_obligations": 0.0,
                    "award_outlays": 0.0,
                    "total_obligations": 0.0,
                    "total_outlays": 0.0,
                },
                "additional": None,
            }

        monkeypatch.setattr(client, "_get", fake_get)
        client.get_disaster_overview()
        assert captured["params"] is None


class TestGetAgencySubAgencyBreakdown:
    # Real response shape from the live API contract's own example
    # (usaspending-api's sub_agency.md, SBA FY2018), not a hand-guessed
    # fixture - nested Office children have no abbreviation field, only
    # the top-level SubAgency does.
    CONTRACT_EXAMPLE_BODY: ClassVar[dict[str, Any]] = {
        "toptier_code": "073",
        "fiscal_year": 2018,
        "page_metadata": {
            "page": 1, "total": 1, "limit": 2, "next": 2, "previous": None,
            "hasNext": True, "hasPrevious": False,
        },
        "results": [
            {
                "name": "Small Business Administration",
                "abbreviation": "SBA",
                "total_obligations": 553748221.72,
                "transaction_count": 14358,
                "new_award_count": 13266,
                "children": [
                    {
                        "name": "OFC OF CAPITAL ACCESS", "code": "737010",
                        "total_obligations": 549195419.92, "transaction_count": 13410, "new_award_count": 12417,
                    },
                    {
                        "name": "OFC OF DISASTER ASSISTANCE", "code": "732990",
                        "total_obligations": 4577429.17, "transaction_count": 943, "new_award_count": 576,
                    },
                ],
            },
        ],
        "messages": [],
    }

    def test_parses_real_response_shape(self, monkeypatch):
        client = USASpendingClient()
        monkeypatch.setattr(client, "_get", lambda path, params=None: self.CONTRACT_EXAMPLE_BODY)
        response = client.get_agency_sub_agency_breakdown("073", fiscal_year=2018)
        assert isinstance(response, AgencySubAgencyResponse)
        assert response.results[0].abbreviation == "SBA"
        assert response.results[0].transaction_count == 14358
        assert response.results[0].children[0].code == "737010"
        assert response.page_metadata.hasNext is True

    def test_none_params_omitted_from_request(self, monkeypatch):
        client = USASpendingClient()
        captured: dict = {}

        def fake_get(path, params=None):
            captured.update(params or {})
            return self.CONTRACT_EXAMPLE_BODY

        monkeypatch.setattr(client, "_get", fake_get)
        client.get_agency_sub_agency_breakdown("073")
        assert "fiscal_year" not in captured
        assert "award_type_codes" not in captured
        assert captured["agency_type"] == "awarding"

    def test_award_type_codes_passed_through(self, monkeypatch):
        client = USASpendingClient()
        captured: dict = {}

        def fake_get(path, params=None):
            captured.update(params or {})
            return self.CONTRACT_EXAMPLE_BODY

        monkeypatch.setattr(client, "_get", fake_get)
        client.get_agency_sub_agency_breakdown("073", award_type_codes=["02", "03", "04", "05"])
        assert captured["award_type_codes"] == ["02", "03", "04", "05"]


class TestAdvancedFilters:
    def test_none_fields_excluded_from_dump(self):
        filters = AdvancedFilters(keywords=["prime award"])
        dumped = filters.model_dump(exclude_none=True)
        assert dumped == {"keywords": ["prime award"]}
        assert "time_period" not in dumped
        assert "agencies" not in dumped

    def test_recipient_id_included_when_set(self):
        # Confirmed live 2026-09-08: precise, reproduces a recipient's
        # true all-time total exactly - only on get_spending_by_category/
        # get_spending_over_time, silently ignored on search_awards (see
        # this field's own docstring on AdvancedFilters).
        filters = AdvancedFilters(recipient_id="419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P")
        dumped = filters.model_dump(exclude_none=True)
        assert dumped == {"recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"}

    def test_extra_fields_are_allowed_and_preserved(self):
        # AdvancedFilters deliberately doesn't model every API filter field
        # (naics_codes, psc_codes, tas_codes, ...) - extras must pass through
        filters = AdvancedFilters(keywords=["test"], naics_codes={"require": ["33"]})
        dumped = filters.model_dump(exclude_none=True)
        assert dumped["naics_codes"] == {"require": ["33"]}


class TestRequestCapture:
    def _fake_response(self, url: str) -> requests.Response:
        resp = make_response(200, {"ok": True})
        resp.url = url
        return resp

    def test_get_records_the_real_resolved_url(self, monkeypatch):
        drain_request_capture()  # discard any leftover from a prior test
        client = USASpendingClient()
        monkeypatch.setattr(
            client.session, "get", lambda url, params=None, timeout=None: self._fake_response(f"{url}?foo=bar")
        )
        client._get("/api/v2/agency/049/", params={"foo": "bar"})
        assert drain_request_capture() == [("GET", "https://api.usaspending.gov/api/v2/agency/049/?foo=bar", None)]

    def test_post_records_method_url_and_body(self, monkeypatch):
        drain_request_capture()
        client = USASpendingClient()
        monkeypatch.setattr(
            client.session, "post", lambda url, json=None, timeout=None: self._fake_response(url)
        )
        client._post("/api/v2/search/spending_by_category/naics/", {"agencies": []})
        assert drain_request_capture() == [
            ("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/naics/", {"agencies": []})
        ]

    def test_drain_clears_the_buffer(self, monkeypatch):
        drain_request_capture()
        client = USASpendingClient()
        monkeypatch.setattr(client.session, "post", lambda url, json=None, timeout=None: self._fake_response(url))
        client._post("/api/v2/search/spending_by_category/naics/", {})
        drain_request_capture()
        assert drain_request_capture() == []

    def test_failed_request_is_still_captured(self, monkeypatch):
        # _record_request runs before _raise_with_detail - a failed call
        # should still show up (the caller decides whether to use it).
        drain_request_capture()
        client = USASpendingClient()
        monkeypatch.setattr(client.session, "get", lambda url, params=None, timeout=None: make_response(404, {}))
        with pytest.raises(USASpendingAPIError):
            client._get("/api/v2/agency/nonexistent/")
        captured = drain_request_capture()
        assert len(captured) == 1
        assert captured[0][0] == "GET"


class TestTimeoutAndConnectionErrorWrapping:
    """_get/_post must turn a raw requests exception (raised when the
    mounted Retry exhausts, or a connection never completes at all) into a
    USASpendingAPIError with an honest, actionable message - not let it
    propagate as an unhandled exception no caller here catches."""

    @pytest.mark.parametrize(
        "exc",
        [
            requests.exceptions.Timeout("timed out"),
            requests.exceptions.ConnectionError("connection refused"),
            requests.exceptions.RetryError("max retries exceeded"),
        ],
    )
    def test_get_wraps_timeout_and_connection_errors(self, monkeypatch, exc):
        client = USASpendingClient()

        def fake_get(url, params=None, timeout=None):
            raise exc

        monkeypatch.setattr(client.session, "get", fake_get)
        with pytest.raises(USASpendingAPIError) as exc_info:
            client._get("/api/v2/agency/049/")
        assert "USASpending.gov" in str(exc_info.value)
        assert exc_info.value.__cause__ is exc

    @pytest.mark.parametrize(
        "exc",
        [
            requests.exceptions.Timeout("timed out"),
            requests.exceptions.ConnectionError("connection refused"),
            requests.exceptions.RetryError("max retries exceeded"),
        ],
    )
    def test_post_wraps_timeout_and_connection_errors(self, monkeypatch, exc):
        client = USASpendingClient()

        def fake_post(url, json=None, timeout=None):
            raise exc

        monkeypatch.setattr(client.session, "post", fake_post)
        with pytest.raises(USASpendingAPIError) as exc_info:
            client._post("/api/v2/search/spending_by_category/naics/", {})
        assert "USASpending.gov" in str(exc_info.value)
        assert exc_info.value.__cause__ is exc

    def test_non_timeout_http_error_path_is_unaffected(self, monkeypatch):
        # A normal 4xx/5xx response (no exception raised by session.get
        # itself) must still go through _raise_with_detail exactly as
        # before - this fix only wraps exceptions from the request call
        # itself, not ordinary error status codes.
        client = USASpendingClient()
        monkeypatch.setattr(client.session, "get", lambda url, params=None, timeout=None: make_response(404, {}))
        with pytest.raises(USASpendingAPIError) as exc_info:
            client._get("/api/v2/agency/nonexistent/")
        assert "404" in str(exc_info.value)

import json

import pytest
import requests

from backend.app.usaspending_client import (
    AdvancedFilters,
    ToptierAgency,
    USASpendingAPIError,
    USASpendingClient,
    _raise_with_detail,
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

    def test_no_match_returns_none(self, client, agencies, monkeypatch):
        monkeypatch.setattr(client, "list_toptier_agencies", lambda: agencies)
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


class TestSearchAwardsValidation:
    def test_raises_without_award_type_codes(self):
        client = USASpendingClient()
        with pytest.raises(ValueError, match="award_type_codes"):
            client.search_awards(AdvancedFilters(keywords=["test"]), fields=["Award ID"])


class TestAdvancedFilters:
    def test_none_fields_excluded_from_dump(self):
        filters = AdvancedFilters(keywords=["prime award"])
        dumped = filters.model_dump(exclude_none=True)
        assert dumped == {"keywords": ["prime award"]}
        assert "time_period" not in dumped
        assert "agencies" not in dumped

    def test_extra_fields_are_allowed_and_preserved(self):
        # AdvancedFilters deliberately doesn't model every API filter field
        # (naics_codes, psc_codes, tas_codes, ...) - extras must pass through
        filters = AdvancedFilters(keywords=["test"], naics_codes={"require": ["33"]})
        dumped = filters.model_dump(exclude_none=True)
        assert dumped["naics_codes"] == {"require": ["33"]}

import json
from types import SimpleNamespace

import pytest

from backend.app.agent import guided_flow
from backend.app.usaspending_client import (
    CategoryResult,
    SpendingByCategoryResponse,
    USASpendingAPIError,
    _record_request,
)


class _FakeClient:
    def __init__(self, spending_by_category):
        self.spending_by_category = spending_by_category


def _fake_text_response(payload: dict):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload))])


@pytest.fixture(autouse=True)
def clear_flow_state():
    guided_flow._FLOW_STATE.clear()
    yield
    guided_flow._FLOW_STATE.clear()


class TestComputeStep:
    def test_missing_fields_returns_collecting(self):
        result = guided_flow.compute_step({"state": "IL"})
        assert result["status"] == "collecting"
        assert result["fields"] == {"state": "IL"}
        assert result.get("prompt") is None

    def test_all_fields_present_returns_separate_prime_and_subaward_totals(self, monkeypatch):
        # place_of_performance_locations' district_current is confirmed
        # live to correctly restrict results to exactly the requested
        # district (unlike recipient_locations - see this module's
        # docstring and fedspendingtransparency/usaspending-api#2372), so a
        # single-row-per-district fake is realistic here.
        def fake_spending_by_category(category, filters, limit=10, spending_level="transactions"):
            assert filters.place_of_performance_locations[0].district_current == "01"
            assert filters.recipient_locations is None
            _record_request("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/district/", {})
            if category == "recipient":
                name = "SUBRECIPIENT CO" if spending_level == "subawards" else "PRIME CO"
                return SpendingByCategoryResponse(
                    category="recipient", limit=limit, results=[CategoryResult(name=name, code=None, amount=1.0)]
                )
            amount = 728235.0 if spending_level == "subawards" else 4_200_000.0
            return SpendingByCategoryResponse(
                category="district", limit=limit, results=[CategoryResult(name="IL-01", code="01", amount=amount)]
            )

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "IL", "district": "01", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["status"] == "result"
        assert result["prime_total"] == 4_200_000.0
        assert result["subaward_total"] == 728235.0
        assert result["top_recipients"] == [{"name": "PRIME CO", "amount": 1.0}]
        assert result["top_subrecipients"] == [{"name": "SUBRECIPIENT CO", "amount": 1.0}]
        # 2 for the district totals, 2 for the top-recipients/subrecipients
        # lookups.
        assert len(result["tool_citations"]) == 4
        # Regression test: confirmed live 2026-09-17 the *first* raw call's
        # request never made it into its citation's curl - a @traceable
        # contextvars propagation quirk (see compute_step's comment on the
        # drain_request_capture() priming call) that a bare citation-count
        # assertion doesn't catch, since build_tool_citation still returns a
        # citation object with curl=None rather than failing outright.
        assert all(c.curl for c in result["tool_citations"])
        assert result.get("note") is None

    def test_at_large_district_retries_with_00_and_discloses_it(self, monkeypatch):
        # Regression test: confirmed live 2026-09-17 that single-district
        # states (e.g. Wyoming) use district "00" (at-large), not a number
        # like "01" - the live API doesn't error on a wrong district, it
        # silently returns empty results, which would otherwise be reported
        # as a confident but wrong "$0".
        def fake_spending_by_category(category, filters, limit=10, spending_level="transactions"):
            district = filters.place_of_performance_locations[0].district_current
            _record_request("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/district/", {})
            if district != "00":
                return SpendingByCategoryResponse(category=category, limit=limit, results=[])
            amount = 100.0 if spending_level == "subawards" else 6_000_000_000.0
            return SpendingByCategoryResponse(
                category=category, limit=limit, results=[CategoryResult(name="WY-00", code="00", amount=amount)]
            )

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "WY", "district": "01", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["status"] == "result"
        assert result["prime_total"] == 6_000_000_000.0
        assert result["subaward_total"] == 100.0
        assert result["fields"]["district"] == "00"
        assert "at-large" in result["note"]

    def test_genuinely_empty_result_is_not_mistaken_for_at_large(self, monkeypatch):
        # District "00" itself also returning nothing means this really is
        # a zero-data case, not an at-large mismatch - must not fabricate a
        # note.
        def fake_spending_by_category(category, filters, limit=10, spending_level="transactions"):
            _record_request("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/district/", {})
            return SpendingByCategoryResponse(category=category, limit=limit, results=[])

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "IL", "district": "01", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["status"] == "result"
        assert result["prime_total"] == 0.0
        assert result["subaward_total"] == 0.0
        assert result["fields"]["district"] == "01"
        assert result.get("note") is None

    def test_api_error_returns_collecting_with_message(self, monkeypatch):
        def fake_spending_by_category(*a, **kw):
            raise USASpendingAPIError("bad district")

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "IL", "district": "99", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["status"] == "collecting"
        assert "bad district" in result["prompt"]

    def test_al_alias_normalizes_to_00_before_ever_reaching_the_api(self, monkeypatch):
        seen_districts = []

        def fake_spending_by_category(category, filters, limit=10, spending_level="transactions"):
            seen_districts.append(filters.place_of_performance_locations[0].district_current)
            _record_request("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/district/", {})
            amount = 1.0 if category == "district" else 0.5
            return SpendingByCategoryResponse(
                category=category, limit=limit, results=[CategoryResult(name="WY-00", code="00", amount=amount)]
            )

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "WY", "district": "al", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["fields"]["district"] == "00"
        assert result["prime_total"] == 1.0
        # Never sent "al" or "AL" to the live API - normalized before the
        # first call, not left to the empty-result retry path to fix.
        assert all(d == "00" for d in seen_districts)

    def test_blank_district_returns_state_breakdown(self, monkeypatch):
        def fake_spending_by_category(category, filters, limit=10, spending_level="transactions"):
            assert filters.place_of_performance_locations[0].district_current is None
            _record_request("POST", "https://api.usaspending.gov/api/v2/search/spending_by_category/district/", {})
            amount = 1_000.0 if spending_level == "subawards" else 9_000.0
            return SpendingByCategoryResponse(
                category="district", limit=limit,
                results=[
                    CategoryResult(name="IL-01", code="01", amount=amount),
                    CategoryResult(name="IL-02", code="02", amount=amount / 2),
                ],
            )

        monkeypatch.setattr(
            "backend.app.agent.tools.spending._get_usaspending_client",
            lambda: _FakeClient(fake_spending_by_category),
        )

        fields = {"state": "IL", "start_fiscal_year": 2023, "end_fiscal_year": 2023}
        result = guided_flow.compute_step(fields)

        assert result["status"] == "breakdown"
        assert len(result["districts"]) == 2
        assert result["districts"][0]["code"] == "01"
        assert result["state_prime_total"] == 9_000.0 + 4_500.0
        assert result["state_subaward_total"] == 1_000.0 + 500.0


class TestClassifyFlowInput:
    def test_valid_json_is_returned(self, monkeypatch):
        payload = {"intent": "slot_update", "fields": {"state": "IL", "district": "01"}, "aside_topic": None}
        monkeypatch.setattr(
            "backend.app.agent.guided_flow._get_client",
            lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _fake_text_response(payload))),
        )
        result = guided_flow.classify_flow_input("IL-01", {})
        assert result == payload

    def test_markdown_fenced_json_is_still_parsed(self, monkeypatch):
        # Regression test: confirmed live 2026-09-17 the model wraps its
        # response in a ```json fence despite the prompt saying "ONLY a
        # JSON object, no other text" - a naive json.loads(text) on the raw
        # response fails and silently defaulted every real slot_update to
        # "escape", breaking the flow on its very first real submission.
        payload = {"intent": "slot_update", "fields": {"state": "IL", "district": "01"}, "aside_topic": None}
        fenced = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=f"```json\n{json.dumps(payload)}\n```")]
        )
        monkeypatch.setattr(
            "backend.app.agent.guided_flow._get_client",
            lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: fenced)),
        )
        result = guided_flow.classify_flow_input("State IL, congressional district 01", {})
        assert result == payload

    def test_malformed_json_defaults_to_escape(self, monkeypatch):
        response = SimpleNamespace(content=[SimpleNamespace(type="text", text="not json")])
        monkeypatch.setattr(
            "backend.app.agent.guided_flow._get_client",
            lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: response)),
        )
        result = guided_flow.classify_flow_input("anything", {})
        assert result["intent"] == "escape"

    def test_unrecognized_intent_defaults_to_escape(self, monkeypatch):
        payload = {"intent": "do_something_else", "fields": {}, "aside_topic": None}
        monkeypatch.setattr(
            "backend.app.agent.guided_flow._get_client",
            lambda: SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: _fake_text_response(payload))),
        )
        result = guided_flow.classify_flow_input("anything", {})
        assert result["intent"] == "escape"


class TestStep:
    def test_start_flow_with_no_state_returns_collecting(self):
        result = guided_flow.step("conv-1", None)
        assert result["status"] == "collecting"
        assert guided_flow._FLOW_STATE["conv-1"]["fields"] == {}

    def test_slot_update_fills_fields_and_recomputes(self, monkeypatch):
        guided_flow.start_flow("conv-1")
        monkeypatch.setattr(
            "backend.app.agent.guided_flow.classify_flow_input",
            lambda message, fields: {
                "intent": "slot_update",
                "fields": {"state": "IL", "district": "01"},
                "aside_topic": None,
            },
        )
        result = guided_flow.step("conv-1", "IL-01")
        assert result["status"] == "collecting"
        assert result["fields"] == {"state": "IL", "district": "01"}
        assert result.get("prompt") is None

    def test_aside_answers_without_clearing_fields(self, monkeypatch):
        guided_flow.start_flow("conv-1")
        guided_flow._FLOW_STATE["conv-1"]["fields"]["state"] = "IL"
        monkeypatch.setattr(
            "backend.app.agent.guided_flow.classify_flow_input",
            lambda message, fields: {"intent": "aside", "fields": {}, "aside_topic": "subaward"},
        )
        result = guided_flow.step("conv-1", "what's a subaward?")
        assert result["status"] == "aside_answered"
        assert "double-count" in result["answer"]
        assert guided_flow._FLOW_STATE["conv-1"]["fields"] == {"state": "IL"}

    def test_escape_pauses_flow_and_forwards_question(self, monkeypatch):
        guided_flow.start_flow("conv-1")
        monkeypatch.setattr(
            "backend.app.agent.guided_flow.classify_flow_input",
            lambda message, fields: {"intent": "escape", "fields": {}, "aside_topic": None},
        )
        result = guided_flow.step("conv-1", "what's NASA's budget?")
        assert result == {"status": "escaped", "forward_question": "what's NASA's budget?"}
        assert guided_flow._FLOW_STATE["conv-1"]["paused"] is True

    def test_resume_after_escape_recomputes_from_held_fields(self, monkeypatch):
        guided_flow.start_flow("conv-1")
        guided_flow._FLOW_STATE["conv-1"]["fields"] = {"state": "IL"}
        guided_flow._FLOW_STATE["conv-1"]["paused"] = True

        result = guided_flow.step("conv-1", None)

        assert result["status"] == "collecting"
        assert result["fields"] == {"state": "IL"}
        assert guided_flow._FLOW_STATE["conv-1"]["paused"] is False

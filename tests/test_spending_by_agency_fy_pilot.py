from types import SimpleNamespace
from unittest.mock import patch

from backend.app.agent.spending_by_agency_fy_pilot import (
    AgencyFYSpendingIntent,
    _extract_agency_fy_spending_intent,
    _looks_like_agency_fy_spending_request,
    handle_agency_fy_spending_request,
)
from backend.app.usaspending import (
    SpendingExplorerResponse,
    SpendingExplorerResult,
    USASpendingAPIError,
)

_MODULE = "backend.app.agent.spending_by_agency_fy_pilot"


def make_text_response(text="NSF spent $1,000."):
    return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])


class TestLooksLikeAgencyFYSpendingRequest:
    def test_matches_spend_verb_and_year(self):
        assert _looks_like_agency_fy_spending_request("How much did NSF spend in FY2024?")

    def test_matches_obligated_phrasing(self):
        assert _looks_like_agency_fy_spending_request("What did HHS obligate in 2023?")

    def test_no_year_does_not_match(self):
        assert not _looks_like_agency_fy_spending_request("How much did NSF spend?")

    def test_no_spend_verb_does_not_match(self):
        assert not _looks_like_agency_fy_spending_request("What awards did NSF make in FY2024?")


def make_tool_use_response(**intent_fields):
    return SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", input=intent_fields)]
    )


class TestExtractAgencyFYSpendingIntent:
    """Regression coverage for a live eval finding (issue #233's before/after
    eval run): the model reliably filled agency_raw/fiscal_year even on
    multi-agency/other-filter questions when the schema only asked it to
    *omit* those fields, so other_filters_present is a required, explicit
    judgment call instead."""

    def test_other_filters_present_true_falls_through(self):
        response = make_tool_use_response(other_filters_present=True, agency_raw="NASA", fiscal_year=2023)
        with patch(f"{_MODULE}._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            intent = _extract_agency_fy_spending_intent("Compare NASA's and DOE's spending in FY2023.")
        assert intent is None

    def test_other_filters_present_false_returns_intent(self):
        response = make_tool_use_response(other_filters_present=False, agency_raw="NSF", fiscal_year=2024)
        with patch(f"{_MODULE}._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            intent = _extract_agency_fy_spending_intent("How much did NSF spend in FY2024?")
        assert intent is not None
        assert intent.agency_raw == "NSF"
        assert intent.fiscal_year == 2024


class TestHandleAgencyFYSpendingRequest:
    def test_ambiguous_intent_falls_through_to_tool_loop(self):
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=None):
            result = handle_agency_fy_spending_request("How much did NSF and HHS spend in FY2024?", "conv-1")
        assert result is None

    def test_unresolvable_agency_falls_through_to_tool_loop(self):
        intent = AgencyFYSpendingIntent(agency_raw="Not A Real Agency", fiscal_year=2024)
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=intent), \
             patch(
                 f"{_MODULE}.get_spending_explorer_breakdown_raw",
                 side_effect=USASpendingAPIError("No agency found matching 'Not A Real Agency'"),
             ):
            result = handle_agency_fy_spending_request("How much did Not A Real Agency spend in FY2024?", "conv-1")
        assert result is None

    def test_incomplete_fiscal_year_falls_through_to_tool_loop(self):
        intent = AgencyFYSpendingIntent(agency_raw="NSF", fiscal_year=2026)
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=intent), \
             patch(
                 f"{_MODULE}.get_spending_explorer_breakdown_raw",
                 side_effect=USASpendingAPIError("Fiscal parameters provided do not belong to a current submission period"),
             ):
            result = handle_agency_fy_spending_request("How much did NSF spend in FY2026?", "conv-1")
        assert result is None

    def test_confident_match_returns_agent_result(self):
        intent = AgencyFYSpendingIntent(agency_raw="NSF", fiscal_year=2024)
        response = SpendingExplorerResponse(
            total=1_000_000.0,
            end_date="2024-09-30",
            results=[
                SpendingExplorerResult(id="806", code="4900", type="agency", name="NSF", amount=1_000_000.0),
            ],
        )
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=intent), \
             patch(f"{_MODULE}.get_spending_explorer_breakdown_raw", return_value=response), \
             patch(f"{_MODULE}._get_client") as get_client:
            get_client.return_value.messages.create.return_value = make_text_response()
            result = handle_agency_fy_spending_request("How much did NSF spend in FY2024?", "conv-1")
        assert result is not None
        assert result.answer_text == "NSF spent $1,000."
        assert result.charts == []
        assert len(result.tool_citations) == 1
        assert result.tool_citations[0].tool_name == "get_spending_explorer_breakdown"

    def test_object_class_breakdown_produces_chart(self):
        intent = AgencyFYSpendingIntent(agency_raw="NSF", fiscal_year=2024, group_by="object_class")
        response = SpendingExplorerResponse(
            total=1_000_000.0,
            end_date="2024-09-30",
            results=[
                SpendingExplorerResult(id="10", code="10", type="object_class", name="Personnel", amount=600_000.0),
                SpendingExplorerResult(id="20", code="20", type="object_class", name="Contracts", amount=400_000.0),
            ],
        )
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=intent), \
             patch(f"{_MODULE}.get_spending_explorer_breakdown_raw", return_value=response) as raw_call, \
             patch(f"{_MODULE}._get_client") as get_client:
            get_client.return_value.messages.create.return_value = make_text_response()
            result = handle_agency_fy_spending_request("How much did NSF spend by object class in FY2024?", "conv-1")
        assert result is not None
        assert len(result.charts) == 1
        assert raw_call.call_args.args[0] == "object_class"

    def test_no_results_falls_through_to_tool_loop(self):
        intent = AgencyFYSpendingIntent(agency_raw="NSF", fiscal_year=2024)
        response = SpendingExplorerResponse(total=None, end_date="2024-09-30", results=[])
        with patch(f"{_MODULE}._extract_agency_fy_spending_intent", return_value=intent), \
             patch(f"{_MODULE}.get_spending_explorer_breakdown_raw", return_value=response):
            result = handle_agency_fy_spending_request("How much did NSF spend in FY2024?", "conv-1")
        assert result is None

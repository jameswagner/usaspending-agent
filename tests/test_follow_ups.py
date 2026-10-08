"""follow_ups_for (response_shaping.py) - the "Download this" follow-up button's
eligibility/filter-carryover logic. Same unit-test shape as should_chart's own
coverage in test_agent.py: pure function, no LLM/network calls."""
from backend.app.agent.response_shaping import FollowUp, follow_ups_for


class TestFollowUpGating:
    def test_search_awards_with_time_period_offers_download(self):
        context = {"agency_name": "National Science Foundation", "start_year": 2024, "end_year": 2024, "award_type": "contracts"}
        follow_ups = follow_ups_for("search_awards", None, context)
        assert len(follow_ups) == 1
        assert follow_ups[0] == FollowUp(
            kind="download",
            label="Download this data as a CSV",
            filters={
                "start_year": 2024, "end_year": 2024, "agency_raw": "National Science Foundation",
                "award_type": "contracts", "spending_level": "awards",
            },
        )

    def test_agency_budget_tools_never_offer_download(self):
        """The most important exclusion: budgetary-resources data has no award-level
        CSV equivalent - a button there would hand back award numbers that don't
        match what's on screen."""
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("get_agency_budget", None, context) == []
        assert follow_ups_for("get_agency_budget_by_subcomponent", None, context) == []

    def test_tool_not_in_allowlist_never_offers_download(self):
        context = {"start_year": 2024, "end_year": 2024}
        assert follow_ups_for("get_award_details", None, context) == []
        assert follow_ups_for("search_recipients", None, context) == []

    def test_no_context_offers_nothing(self):
        assert follow_ups_for("search_awards", None, None) == []
        assert follow_ups_for("search_awards", None, {}) == []

    def test_missing_time_period_offers_nothing(self):
        """Gate per the design: a real time period is required - a tool call
        resolved with no start_year/end_year shouldn't offer a download button."""
        context = {"agency_name": "NSF"}
        assert follow_ups_for("search_awards", None, context) == []


class TestFollowUpFilterCarryover:
    def test_naics_scoped_category_call_carries_naics_code(self):
        """Replayed for the button: a NAICS-scoped answer's
        follow-up must carry naics_code, not just agency/time_period."""
        context = {
            "agency_name": "National Science Foundation", "start_year": 2024, "end_year": 2024,
            "naics_code": "518210", "category": "naics", "spending_level": "transactions",
        }
        follow_ups = follow_ups_for("get_spending_by_category", None, context)
        assert follow_ups[0].filters["naics_code"] == "518210"
        # Display-only args must never leak in as scope filters.
        assert "category" not in follow_ups[0].filters

    def test_def_codes_list_survives_structured_not_flattened(self):
        """ToolCitation.parameters comma-joins list filters for
        display - the follow-up's own filters dict must NOT do that, since it's
        posted straight back as a real DownloadIntent field."""
        context = {
            "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "def_codes": ["L", "M"], "spending_level": "transactions",
        }
        follow_ups = follow_ups_for("get_spending_over_time", None, context)
        assert follow_ups[0].filters["def_codes"] == ["L", "M"]

    def test_display_only_args_excluded(self):
        context = {
            "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "category": "naics", "spending_level": "transactions",
        }
        follow_ups = follow_ups_for("get_spending_by_category", None, context)
        assert "category" not in follow_ups[0].filters

    def test_geography_axis_args_excluded(self):
        context = {
            "agency_name": "NSF", "start_year": 2024, "end_year": 2024,
            "scope": "place_of_performance", "geo_layer": "state", "sort_by": "amount",
        }
        follow_ups = follow_ups_for("get_spending_by_geography", None, context)
        filters = follow_ups[0].filters
        assert "scope" not in filters
        assert "geo_layer" not in filters
        assert "sort_by" not in filters

    def test_recipient_id_only_scope_offers_no_download(self):
        """recipient_id has no /download/search/ equivalent - if it's the
        prior call's ONLY scope, a button here would silently download unscoped
        data instead of what was shown, which is worse than no button."""
        context = {"recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("get_spending_over_time", None, context) == []

    def test_recipient_id_alongside_other_scope_still_offers_download_without_it(self):
        context = {
            "recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", "naics_code": "518210",
            "start_year": 2024, "end_year": 2024,
        }
        follow_ups = follow_ups_for("get_spending_over_time", None, context)
        assert len(follow_ups) == 1
        assert "recipient_id" not in follow_ups[0].filters
        assert follow_ups[0].filters["naics_code"] == "518210"


class TestFollowUpSpendingLevel:
    def test_search_awards_fixed_to_awards(self):
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("search_awards", None, context)[0].filters["spending_level"] == "awards"

    def test_search_subawards_fixed_to_subawards(self):
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("search_subawards", None, context)[0].filters["spending_level"] == "subawards"

    def test_search_transactions_fixed_to_transactions(self):
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("search_transactions", None, context)[0].filters["spending_level"] == "transactions"

    def test_category_tool_uses_context_spending_level_not_download_intent_default(self):
        """DownloadIntent itself defaults spending_level to "awards", but
        the spending tools' own citations default to "transactions" - a naive
        pass-through would silently flip the level relative to what was shown."""
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024, "spending_level": "transactions"}
        assert follow_ups_for("get_spending_by_category", None, context)[0].filters["spending_level"] == "transactions"

    def test_award_type_breakdown_defaults_to_transactions(self):
        """No spending_level parameter exists on this tool at all - the same
        "transactions" default used elsewhere stands in rather than DownloadIntent's
        own "awards" default."""
        context = {"agency_name": "NSF", "start_year": 2024, "end_year": 2024}
        assert follow_ups_for("get_award_type_breakdown", None, context)[0].filters["spending_level"] == "transactions"

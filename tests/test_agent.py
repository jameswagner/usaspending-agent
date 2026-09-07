from datetime import date, datetime, timezone

from backend.app.agent.response_shaping import (
    build_tool_citation,
    current_fiscal_year,
    fiscal_year_to_date_range,
    should_chart,
)
from backend.app.agent.tools import AWARD_TYPE_GROUPS, _normalize_award_type
from backend.app.usaspending_client import (
    CategoryResult,
    SpendingByCategoryResponse,
    SpendingOverTimeResponse,
    TimePeriodGroup,
    TimeResult,
)


def make_category_response(n: int) -> SpendingByCategoryResponse:
    return SpendingByCategoryResponse(
        category="naics",
        limit=10,
        results=[
            CategoryResult(name=f"Category {i}", code=str(i), amount=float(i) * 100)
            for i in range(n)
        ],
    )


def make_time_response(n: int) -> SpendingOverTimeResponse:
    return SpendingOverTimeResponse(
        group="fiscal_year",
        results=[
            TimeResult(
                time_period=TimePeriodGroup(fiscal_year=str(2020 + i)),
                aggregated_amount=float(i) * 1000,
            )
            for i in range(n)
        ],
    )


class TestCurrentFiscalYear:
    # Regression coverage for the real bug: asked for NSF's "most recent
    # fiscal year" NAICS breakdown, the model answered FY2024 while FY2025
    # data was already live, because nothing told it what today's date
    # actually is. These pin exact dates across the Oct 1 rollover so the
    # boundary itself is asserted, not just "some plausible year."

    def test_mid_fiscal_year(self):
        # Sep 4 2026 falls in FY2026 (Oct 2025-Sep 2026).
        assert current_fiscal_year(date(2026, 9, 4)) == 2026

    def test_day_before_rollover_is_still_prior_fy(self):
        assert current_fiscal_year(date(2025, 9, 30)) == 2025

    def test_rollover_day_is_next_fy(self):
        assert current_fiscal_year(date(2025, 10, 1)) == 2026

    def test_defaults_to_the_real_current_date(self):
        assert current_fiscal_year() == current_fiscal_year(datetime.now(timezone.utc).date())


class TestFiscalYearToDateRange:
    def test_single_fiscal_year(self):
        # FY2021 runs Oct 2020 - Sep 2021 - named by the year it ENDS in.
        assert fiscal_year_to_date_range(2021, 2021) == ("2020-10-01", "2021-09-30")

    def test_multi_year_range_regression(self):
        # Direct regression test for the real bug: asked for "2021 to 2024,"
        # the model (before this fix existed) computed start_date=2021-10-01
        # itself - the start of FY2022, not FY2021. This asserts the code-
        # computed start date is the one FY2021 actually begins on.
        start, end = fiscal_year_to_date_range(2021, 2024)
        assert start == "2020-10-01"
        assert end == "2024-09-30"

    def test_start_is_not_the_bug_off_by_one_value(self):
        start, _ = fiscal_year_to_date_range(2021, 2024)
        assert start != "2021-10-01"


class TestSpendingOverTime:
    def test_multi_period_produces_line_spec(self):
        spec = should_chart("get_spending_over_time", make_time_response(3))
        assert spec is not None
        assert spec.chart_type == "line"
        assert spec.labels == ["FY2020", "FY2021", "FY2022"]
        assert spec.values == [0.0, 1000.0, 2000.0]

    def test_single_period_returns_none(self):
        assert should_chart("get_spending_over_time", make_time_response(1)) is None

    def test_zero_periods_returns_none(self):
        assert should_chart("get_spending_over_time", make_time_response(0)) is None


class TestSpendingByCategory:
    def test_multi_category_produces_bar_spec(self):
        spec = should_chart("get_spending_by_category", make_category_response(3))
        assert spec is not None
        assert spec.chart_type == "bar"
        assert spec.labels == ["Category 0", "Category 1", "Category 2"]
        assert spec.values == [0.0, 100.0, 200.0]

    def test_single_category_returns_none(self):
        assert should_chart("get_spending_by_category", make_category_response(1)) is None

    def test_zero_categories_returns_none(self):
        assert should_chart("get_spending_by_category", make_category_response(0)) is None


class TestNeverChartTools:
    def test_search_guide_never_charts(self):
        assert should_chart("search_guide", make_category_response(5)) is None

    def test_lookup_agency_never_charts(self):
        assert should_chart("lookup_agency", make_category_response(5)) is None

    def test_search_awards_never_charts(self):
        assert should_chart("search_awards", make_category_response(5)) is None

    def test_unknown_tool_name_returns_none(self):
        assert should_chart("some_future_tool", make_category_response(5)) is None


class TestBuildToolCitation:
    def test_lookup_agency(self):
        citation = build_tool_citation("lookup_agency", {"name": "National Science Foundation"})
        assert citation is not None
        assert citation.tool_name == "lookup_agency"
        assert citation.parameters == {"name": "National Science Foundation"}
        assert citation.description == "Agency lookup: National Science Foundation"

    def test_get_spending_by_category(self):
        citation = build_tool_citation(
            "get_spending_by_category",
            {
                "agency_name": "National Science Foundation",
                "category": "naics",
                "start_fiscal_year": 2023,
                "end_fiscal_year": 2024,
            },
        )
        assert citation is not None
        assert citation.tool_name == "get_spending_by_category"
        assert citation.description == "naics breakdown, National Science Foundation, FY2023-FY2024"

    def test_get_spending_over_time(self):
        citation = build_tool_citation(
            "get_spending_over_time",
            {
                "agency_name": "National Science Foundation",
                "start_fiscal_year": 2021,
                "end_fiscal_year": 2024,
                "group": "fiscal_year",
            },
        )
        assert citation is not None
        assert citation.tool_name == "get_spending_over_time"
        assert citation.description == (
            "Spending over time (fiscal_year), National Science Foundation, FY2021-FY2024"
        )

    def test_search_awards(self):
        citation = build_tool_citation(
            "search_awards",
            {
                "agency_name": "National Science Foundation",
                "start_fiscal_year": 2023,
                "end_fiscal_year": 2023,
                "award_type": "grants",
            },
        )
        assert citation is not None
        assert citation.tool_name == "search_awards"
        assert citation.description == "grants awards search, National Science Foundation, FY2023-FY2023"

    def test_search_guide_returns_none(self):
        # search_guide is cited separately, by chunk id/page - not via
        # build_tool_citation.
        assert build_tool_citation("search_guide", {}) is None

    def test_empty_context_returns_none(self):
        # A failed call that returned before _record_tool_call ran (or any
        # call whose context wasn't populated) shouldn't produce a citation.
        assert build_tool_citation("lookup_agency", {}) is None

    def test_unknown_tool_name_returns_none(self):
        assert build_tool_citation("some_future_tool", {"foo": "bar"}) is None


class TestAwardTypeNormalization:
    # Regression coverage for the real bug: asked for NSF's "cooperative
    # agreements," the model picked award_type="contracts" - not just a
    # broader bucket than asked for, but the flat-out wrong one, since a
    # cooperative agreement isn't a contract. Fixed by exposing the real,
    # specific USASpending sub-types (verified against award_types.md) as
    # their own values, not just the three broad buckets.

    def test_normalizes_case_spacing_and_hyphens(self):
        assert _normalize_award_type("Cooperative Agreement") == "cooperative_agreement"
        assert _normalize_award_type("cooperative-agreement") == "cooperative_agreement"
        assert _normalize_award_type("  BPA Call  ") == "bpa_call"

    def test_broad_buckets_still_present(self):
        assert AWARD_TYPE_GROUPS["contracts"] == ["A", "B", "C", "D"]
        assert AWARD_TYPE_GROUPS["grants"] == ["02", "03", "04", "05"]
        assert AWARD_TYPE_GROUPS["loans"] == ["07", "08"]

    def test_cooperative_agreement_resolves_to_the_correct_single_code(self):
        # Code 05 per USASpending's award_types.md - part of the broader
        # "grants" bucket (02-05), but its own precise code, not the
        # bucket as a whole.
        assert AWARD_TYPE_GROUPS["cooperative_agreement"] == ["05"]

    def test_every_specific_subtype_code_is_a_member_of_its_broad_bucket(self):
        # The specific sub-types should be a strict refinement of the
        # broad buckets, not a disjoint or inconsistent set of codes.
        contract_subtypes = ["bpa_call", "purchase_order", "delivery_order", "definitive_contract"]
        for key in contract_subtypes:
            assert AWARD_TYPE_GROUPS[key][0] in AWARD_TYPE_GROUPS["contracts"]

        grant_subtypes = ["block_grant", "formula_grant", "project_grant", "cooperative_agreement"]
        for key in grant_subtypes:
            assert AWARD_TYPE_GROUPS[key][0] in AWARD_TYPE_GROUPS["grants"]

        loan_subtypes = ["direct_loan", "guaranteed_loan"]
        for key in loan_subtypes:
            assert AWARD_TYPE_GROUPS[key][0] in AWARD_TYPE_GROUPS["loans"]

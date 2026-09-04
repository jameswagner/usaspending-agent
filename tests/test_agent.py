from backend.app.agent import should_chart
from backend.app.tools.usaspending_client import (
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


class TestSpendingOverTime:
    def test_multi_period_produces_line_spec(self):
        spec = should_chart("get_spending_over_time", make_time_response(3))
        assert spec is not None
        assert spec.chart_type == "line"
        assert spec.labels == ["2020", "2021", "2022"]
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

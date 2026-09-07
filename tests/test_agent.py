from datetime import date, datetime, timezone

import pytest

from backend.app.agent.response_shaping import (
    build_tool_citation,
    current_fiscal_year,
    fiscal_year_to_date_range,
    should_chart,
)
from backend.app.agent.tool_filters import (
    AWARD_TYPE_GROUPS,
    LOAN_AWARD_TYPE_CODES,
    US_STATE_ABBREVIATIONS,
    _amount_field_for_award_type,
    _build_filters,
    _normalize_award_type,
    _normalize_state,
)
from backend.app.usaspending_client import (
    CategoryResult,
    SpendingByCategoryResponse,
    SpendingOverTimeResponse,
    TimePeriodGroup,
    TimeResult,
    ToptierAgency,
    USASpendingAPIError,
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

    def test_get_spending_by_category_includes_new_filters_when_present(self):
        citation = build_tool_citation(
            "get_spending_by_category",
            {
                "agency_name": "National Science Foundation",
                "category": "naics",
                "start_fiscal_year": 2023,
                "end_fiscal_year": 2024,
                "award_type": "grants",
                "min_amount": 1_000_000.0,
            },
        )
        assert citation.parameters["award_type"] == "grants"
        assert citation.parameters["min_amount"] == 1_000_000.0
        # description is unchanged by the new filters - not asserted here,
        # already covered by test_get_spending_by_category above.

    def test_omitted_new_filters_dont_appear_in_citation_params(self):
        citation = build_tool_citation(
            "get_spending_by_category",
            {
                "agency_name": "National Science Foundation",
                "category": "naics",
                "start_fiscal_year": 2023,
                "end_fiscal_year": 2024,
            },
        )
        assert "award_type" not in citation.parameters
        assert "min_amount" not in citation.parameters

    def test_search_awards_award_type_not_duplicated_by_optional_merge(self):
        # award_type is always set for search_awards (unlike the other two
        # tools, where it's one of the optional filters) - the optional
        # merge must not clobber or duplicate it.
        citation = build_tool_citation(
            "search_awards",
            {
                "agency_name": "National Science Foundation",
                "start_fiscal_year": 2023,
                "end_fiscal_year": 2023,
                "award_type": "grants",
                "recipient_name": "Leidos",
            },
        )
        assert citation.parameters["award_type"] == "grants"
        assert citation.parameters["recipient_name"] == "Leidos"

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


def make_agency(name: str = "National Science Foundation") -> ToptierAgency:
    return ToptierAgency(
        agency_id=1, agency_name=name, toptier_code="049", abbreviation="NSF", agency_slug="nsf"
    )


class FakeClient:
    """Stands in for USASpendingClient in _build_filters tests - only
    find_agency_by_name is ever called by _build_filters, so that's all
    that needs faking."""

    def __init__(self, agency: ToptierAgency | None):
        self._agency = agency

    def find_agency_by_name(self, name):
        return self._agency


class TestBuildFilters:
    # Regression coverage for the real findings: no amount-based sort, no
    # award_amounts/recipient_search_text/location filters wired to any
    # tool - all verified live against the real USASpending API before
    # this fix (see BACKLOG.md / the shared-filter-layer plan).

    def test_omitting_all_new_params_reproduces_the_original_filter_shape(self):
        # Hard constraint: no new params set must be byte-for-byte
        # identical to what get_spending_by_category_raw etc. built before
        # this consolidation - agencies + time_period only.
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024)
        dumped = filters.model_dump(exclude_none=True)
        assert set(dumped.keys()) == {"agencies", "time_period"}

    def test_unresolved_agency_raises(self):
        with pytest.raises(USASpendingAPIError, match="No agency found"):
            _build_filters(FakeClient(None), "Not A Real Agency", 2021, 2024)

    def test_award_type_resolves_to_codes(self):
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024, award_type="cooperative_agreement"
        )
        assert filters.award_type_codes == ["05"]

    def test_unknown_award_type_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unknown award_type"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, award_type="not_a_type")

    def test_recipient_name_becomes_single_item_list(self):
        # search_filters.md: recipient_search_text is capped at 1 item.
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024, recipient_name="Leidos"
        )
        assert filters.recipient_search_text == ["Leidos"]

    def test_amount_bounds(self):
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024, min_amount=1_000_000_000
        )
        assert filters.award_amounts[0].lower_bound == 1_000_000_000
        assert filters.award_amounts[0].upper_bound is None

    def test_inverted_amount_bounds_raise(self):
        with pytest.raises(USASpendingAPIError, match="must not exceed"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, min_amount=100, max_amount=50)

    def test_performed_and_recipient_state_are_independent_fields(self):
        # Regression for the real finding: these must map to different
        # AdvancedFilters keys, not collapse into one - a ~$16B live
        # discrepancy for DoD/VA (verified 2026-09-06) depends on this
        # staying true.
        filters = _build_filters(
            FakeClient(make_agency()),
            "NSF",
            2021,
            2024,
            performed_in_state="Virginia",
            recipient_in_state="Texas",
        )
        assert filters.place_of_performance_locations[0].state == "VA"
        assert filters.recipient_locations[0].state == "TX"

    def test_unrecognized_state_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized state"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, performed_in_state="Narnia")


class TestNormalizeState:
    def test_full_name_case_and_spacing_insensitive(self):
        assert _normalize_state("Virginia") == "VA"
        assert _normalize_state("virginia") == "VA"
        assert _normalize_state("  Virginia  ") == "VA"

    def test_abbreviation_passthrough_case_insensitive(self):
        assert _normalize_state("va") == "VA"
        assert _normalize_state("VA") == "VA"

    def test_dc_and_territories(self):
        assert _normalize_state("District of Columbia") == "DC"
        assert _normalize_state("Puerto Rico") == "PR"

    def test_every_abbreviation_round_trips(self):
        for full_name, code in US_STATE_ABBREVIATIONS.items():
            assert _normalize_state(full_name) == code
            assert _normalize_state(code) == code

    def test_garbage_raises_with_clear_message(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized state 'Springfield'"):
            _normalize_state("Springfield")


class TestAmountFieldForAwardType:
    # Regression coverage for the real, not-fully-verified-from-the-doc-
    # alone assumption: "Award Amount" isn't valid for loan-type
    # search_awards results per spending_by_award.md's field tables -
    # loans expose "Loan Value" instead. See
    # dev_tools/verify_shared_filters.py for the live confirmation of this.

    def test_contracts_and_grants_use_award_amount(self):
        assert _amount_field_for_award_type("contracts") == "Award Amount"
        assert _amount_field_for_award_type("grants") == "Award Amount"
        assert _amount_field_for_award_type("cooperative_agreement") == "Award Amount"

    def test_loans_use_loan_value(self):
        assert _amount_field_for_award_type("loans") == "Loan Value"
        assert _amount_field_for_award_type("direct_loan") == "Loan Value"
        assert _amount_field_for_award_type("guaranteed_loan") == "Loan Value"

    def test_loan_award_type_codes_match_the_award_types_contract(self):
        # 07 = Direct Loan, 08 = Guaranteed/Insured Loan per award_types.md.
        assert LOAN_AWARD_TYPE_CODES == {"07", "08"}

    def test_unknown_award_type_falls_back_to_award_amount(self):
        # _build_filters already validated award_type earlier in the same
        # call in real usage - this is just the documented fallback
        # behavior for this function specifically, not a new validation path.
        assert _amount_field_for_award_type("not_a_real_type") == "Award Amount"

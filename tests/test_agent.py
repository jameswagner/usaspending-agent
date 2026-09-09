from datetime import date, datetime, timezone
from typing import Any, ClassVar, get_args

import pytest

from backend.app.agent.response_shaping import (
    GLOSSARY_URL_BASE,
    GUIDE_URL,
    Citation,
    _build_guide_citation,
    _extract_guide_question,
    build_tool_citation,
    current_fiscal_year,
    fiscal_year_to_date_range,
    should_chart,
)
from backend.app.agent.tool_filters import (
    AWARD_TYPE_GROUPS,
    LOAN_AWARD_TYPE_CODES,
    MAX_LIMIT,
    RECIPIENT_AWARD_TYPES,
    US_STATE_ABBREVIATIONS,
    VALID_DATE_TYPES,
    AwardType,
    DateType,
    RecipientAwardType,
    Scope,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
    _normalize_award_type,
    _normalize_date_type,
    _normalize_recipient_award_type,
    _normalize_scope,
    _normalize_state,
    _validate_cfda_program,
    _validate_naics_code,
    _validate_psc_code,
)
from backend.app.agent.tools import (
    MAX_TOOL_CALLS_PER_TURN,
    VALID_CATEGORIES,
    VALID_GROUPS,
    Category,
    Group,
    _agency_label,
    _check_tool_call_budget,
    _format_agency_award_breakdown,
    _format_api_messages,
    _format_award_details,
    _format_business_type,
    _format_contract_or_idv,
    _format_financial_assistance,
    _format_geography_result,
    _format_period_breakdown,
    _format_period_of_performance,
    _format_recipient_address,
    _format_recipient_level,
    _format_recipient_listing,
    _format_recipient_overview,
    _format_recipient_state_only,
    _format_top_agencies_by_budget,
    _location_label,
    _normalize_category,
    _normalize_group,
    _scope_label,
    _tool_call_log,
    _truncation_note,
)
from backend.app.usaspending_client import (
    AgencySubAgencyResponse,
    CategoryResult,
    GeographyTypeResult,
    IDVAmountsResponse,
    ObligationByPeriod,
    RecipientListing,
    RecipientLocation,
    RecipientOverview,
    SpendingByCategoryResponse,
    SpendingByGeographyResponse,
    SpendingOverTimeResponse,
    SubAgencyBreakdown,
    SubAgencyOffice,
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


def make_geography_response(n: int, geo_layer: str = "state") -> SpendingByGeographyResponse:
    return SpendingByGeographyResponse(
        scope="place_of_performance",
        geo_layer=geo_layer,
        spending_level="transactions",
        results=[
            GeographyTypeResult(shape_code=f"S{i}", display_name=f"State {i}", aggregated_amount=float(i) * 1000)
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

    def test_year_below_2008_rejected(self):
        with pytest.raises(USASpendingAPIError, match="out of range"):
            fiscal_year_to_date_range(1776, 2024)

    def test_year_far_beyond_current_rejected(self):
        with pytest.raises(USASpendingAPIError, match="out of range"):
            fiscal_year_to_date_range(2021, 9999)

    def test_next_fiscal_year_is_allowed(self):
        next_fy = current_fiscal_year() + 1
        fiscal_year_to_date_range(next_fy, next_fy)

    def test_start_after_end_rejected(self):
        with pytest.raises(USASpendingAPIError, match="after end_fiscal_year"):
            fiscal_year_to_date_range(2025, 2021)

    def test_floor_year_itself_is_allowed(self):
        fiscal_year_to_date_range(2008, 2008)


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


def make_sub_agency_response(n: int) -> AgencySubAgencyResponse:
    return AgencySubAgencyResponse(
        toptier_code="075",
        fiscal_year=2024,
        results=[
            SubAgencyBreakdown(
                name=f"Sub-Agency {i}",
                abbreviation=f"SA{i}",
                total_obligations=float(i * 100),
                transaction_count=i * 10,
                new_award_count=i,
            )
            for i in range(n)
        ],
    )


class TestGetAgencyAwardBreakdownChart:
    def test_multi_sub_agency_produces_bar_spec(self):
        spec = should_chart("get_agency_award_breakdown", make_sub_agency_response(3))
        assert spec is not None
        assert spec.chart_type == "bar"
        assert spec.labels == ["SA0", "SA1", "SA2"]
        assert spec.values == [0.0, 100.0, 200.0]

    def test_single_sub_agency_returns_none(self):
        assert should_chart("get_agency_award_breakdown", make_sub_agency_response(1)) is None

    def test_missing_abbreviation_falls_back_to_name(self):
        response = AgencySubAgencyResponse(
            toptier_code="075",
            fiscal_year=2024,
            results=[
                SubAgencyBreakdown(name="No Abbreviation Office", abbreviation=None, total_obligations=1.0, transaction_count=1, new_award_count=1),
                SubAgencyBreakdown(name="Other Office", abbreviation="OO", total_obligations=2.0, transaction_count=2, new_award_count=2),
            ],
        )
        spec = should_chart("get_agency_award_breakdown", response)
        assert "No Abbreviation Office" in spec.labels


class TestFormatAgencyAwardBreakdown:
    def test_formats_amount_and_counts(self):
        result = _format_agency_award_breakdown(make_sub_agency_response(1))
        assert "SA0" in result
        assert "$0.00" in result
        assert "0 transactions" in result
        assert "0 new awards" in result

    def test_omits_parens_when_no_abbreviation(self):
        response = AgencySubAgencyResponse(
            toptier_code="075",
            fiscal_year=2024,
            results=[SubAgencyBreakdown(name="Solo Office", abbreviation=None, total_obligations=5.0, transaction_count=2, new_award_count=1)],
        )
        result = _format_agency_award_breakdown(response)
        assert "Solo Office:" in result
        assert "()" not in result

    def test_children_are_not_shown(self):
        response = AgencySubAgencyResponse(
            toptier_code="075",
            fiscal_year=2024,
            results=[
                SubAgencyBreakdown(
                    name="Parent Office", abbreviation="PO", total_obligations=5.0, transaction_count=2, new_award_count=1,
                    children=[SubAgencyOffice(name="Child Office", code="123", total_obligations=5.0, transaction_count=2, new_award_count=1)],
                ),
            ],
        )
        result = _format_agency_award_breakdown(response)
        assert "Child Office" not in result


class TestSpendingByGeographyChart:
    def test_multi_region_produces_bar_spec(self):
        spec = should_chart("get_spending_by_geography", make_geography_response(3))
        assert spec is not None
        assert spec.chart_type == "bar"
        assert spec.labels == ["State 2", "State 1", "State 0"]
        assert spec.values == [2000.0, 1000.0, 0.0]

    def test_single_region_returns_none(self):
        assert should_chart("get_spending_by_geography", make_geography_response(1)) is None

    def test_caps_at_20_regions_sorted_by_amount(self):
        spec = should_chart("get_spending_by_geography", make_geography_response(30))
        assert len(spec.labels) == 20
        assert spec.labels[0] == "State 29"  # highest amount first

    def test_null_display_name_falls_back_to_shape_code(self):
        response = SpendingByGeographyResponse(
            scope="place_of_performance", geo_layer="state", spending_level="transactions",
            results=[
                GeographyTypeResult(shape_code="CA", display_name="California", aggregated_amount=200.0),
                GeographyTypeResult(shape_code=None, display_name=None, aggregated_amount=100.0),
            ],
        )
        spec = should_chart("get_spending_by_geography", response)
        assert "Unknown" in spec.labels


class TestNeverChartTools:
    def test_search_guide_never_charts(self):
        assert should_chart("search_guide", make_category_response(5)) is None

    def test_lookup_agency_never_charts(self):
        assert should_chart("lookup_agency", make_category_response(5)) is None

    def test_search_awards_never_charts(self):
        assert should_chart("search_awards", make_category_response(5)) is None

    def test_get_agency_budget_never_charts(self):
        assert should_chart("get_agency_budget", make_category_response(5)) is None

    def test_get_award_details_never_charts(self):
        # A single award record, not a list - no cardinality to chart.
        assert should_chart("get_award_details", make_category_response(5)) is None

    def test_search_recipients_never_charts(self):
        # Charting a candidate list is plausible in principle but a real,
        # deliberately deferred design question, not resolved here.
        assert should_chart("search_recipients", make_category_response(5)) is None

    def test_get_recipient_details_never_charts(self):
        assert should_chart("get_recipient_details", make_category_response(5)) is None

    def test_unknown_tool_name_returns_none(self):
        assert should_chart("some_future_tool", make_category_response(5)) is None


class TestFormatPeriodBreakdown:
    # Real gap found live (2026-09-07): agency_obligation_by_period was
    # already parsed by the client but never surfaced in get_agency_budget's
    # output at all. Verified live against NSF that each period's amount is
    # CUMULATIVE from the start of the fiscal year (period 12's value
    # exactly equals the year's agency_total_obligated) - these tests pin
    # both the month mapping and the cumulative framing so neither
    # regresses silently.

    def test_periods_map_to_the_right_fiscal_months(self):
        # P01 = October (verified against the local Glossary's "Submission
        # Period" entry), not calendar January.
        periods = [ObligationByPeriod(period=1, obligated=100.0), ObligationByPeriod(period=12, obligated=900.0)]
        result = _format_period_breakdown(periods)
        assert "Oct $100.00" in result
        assert "Sep $900.00" in result

    def test_out_of_order_periods_are_sorted(self):
        periods = [ObligationByPeriod(period=3, obligated=300.0), ObligationByPeriod(period=2, obligated=200.0)]
        result = _format_period_breakdown(periods)
        assert result.index("Nov") < result.index("Dec")

    def test_label_states_cumulative_explicitly(self):
        # The framing that matters: without this, "period 6: $X" reads like
        # a monthly figure, not a running total - the same misleading shape
        # as the Award Amount cumulative bug found earlier in this project.
        result = _format_period_breakdown([ObligationByPeriod(period=1, obligated=100.0)])
        assert "cumulative" in result.lower()


class TestBuildToolCitation:
    def test_get_agency_budget(self):
        citation = build_tool_citation(
            "get_agency_budget",
            {"agency_name": "National Science Foundation", "start_fiscal_year": 2021, "end_fiscal_year": 2024},
        )
        assert citation is not None
        assert citation.tool_name == "get_agency_budget"
        assert citation.description == (
            "Budgetary resources, National Science Foundation, FY2021-FY2024"
        )

    def test_get_agency_award_breakdown(self):
        citation = build_tool_citation(
            "get_agency_award_breakdown",
            {"agency_name": "National Science Foundation", "fiscal_year": 2024, "award_type": "grants"},
        )
        assert citation is not None
        assert citation.tool_name == "get_agency_award_breakdown"
        assert citation.parameters == {
            "agency_name": "National Science Foundation",
            "fiscal_year": 2024,
            "award_type": "grants",
        }
        assert citation.description == "Award breakdown by sub-agency, National Science Foundation, FY2024"

    def test_get_agency_award_breakdown_omits_award_type_when_not_set(self):
        citation = build_tool_citation(
            "get_agency_award_breakdown",
            {"agency_name": "National Science Foundation", "fiscal_year": 2024, "award_type": None},
        )
        assert "award_type" not in citation.parameters

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

    def test_get_award_details(self):
        citation = build_tool_citation(
            "get_award_details",
            {"award_id": "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-", "piid": "NSFDACS1219442"},
        )
        assert citation is not None
        assert citation.tool_name == "get_award_details"
        assert citation.parameters == {"award_id": "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-"}
        # The human-readable piid, not the raw internal_id hash, drives the
        # shown description - a citation showing the ugly hash string would
        # look broken next to every other tool's readable description.
        assert citation.description == "Award details: NSFDACS1219442"

    def test_get_award_details_falls_back_to_award_id_without_piid(self):
        citation = build_tool_citation("get_award_details", {"award_id": "CONT_AWD_X"})
        assert citation.description == "Award details: CONT_AWD_X"

    def test_search_recipients(self):
        citation = build_tool_citation("search_recipients", {"keyword": "Boeing"})
        assert citation is not None
        assert citation.tool_name == "search_recipients"
        assert citation.parameters == {"keyword": "Boeing"}
        assert citation.description == "Recipient search: Boeing"

    def test_get_recipient_details(self):
        citation = build_tool_citation(
            "get_recipient_details",
            {"recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", "name": "THE BOEING COMPANY"},
        )
        assert citation is not None
        assert citation.tool_name == "get_recipient_details"
        assert citation.parameters == {"recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"}
        assert citation.description == "Recipient details: THE BOEING COMPANY"

    def test_get_recipient_details_falls_back_to_recipient_id_without_name(self):
        citation = build_tool_citation("get_recipient_details", {"recipient_id": "abc-P"})
        assert citation.description == "Recipient details: abc-P"

    # agency_name is now optional on all three spending tools (2026-09-08) -
    # these three regression tests pin that the citation builder doesn't
    # KeyError when it's absent, and that the description falls back to
    # whatever scope (recipient_name/recipient_id) was actually given.

    def test_get_spending_by_category_with_recipient_id_no_agency_name(self):
        citation = build_tool_citation(
            "get_spending_by_category",
            {
                "category": "awarding_agency",
                "start_fiscal_year": 2008,
                "end_fiscal_year": 2025,
                "recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P",
            },
        )
        assert citation is not None
        assert "agency_name" not in citation.parameters
        assert citation.description == (
            "awarding_agency breakdown, 419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P, FY2008-FY2025"
        )

    def test_get_spending_over_time_with_recipient_name_no_agency_name(self):
        citation = build_tool_citation(
            "get_spending_over_time",
            {"start_fiscal_year": 2020, "end_fiscal_year": 2024, "group": "fiscal_year", "recipient_name": "Boeing"},
        )
        assert citation.description == "Spending over time (fiscal_year), Boeing, FY2020-FY2024"

    def test_search_awards_with_recipient_name_no_agency_name(self):
        citation = build_tool_citation(
            "search_awards",
            {"start_fiscal_year": 2023, "end_fiscal_year": 2023, "award_type": "contracts", "recipient_name": "Boeing"},
        )
        assert citation.description == "contracts awards search, Boeing, FY2023-FY2023"

    def test_get_spending_by_geography(self):
        citation = build_tool_citation(
            "get_spending_by_geography",
            {
                "scope": "place_of_performance", "geo_layer": "state",
                "start_fiscal_year": 2023, "end_fiscal_year": 2023,
                "agency_name": "National Science Foundation",
            },
        )
        assert citation is not None
        assert citation.description == "Spending by state (place_of_performance), National Science Foundation, FY2023-FY2023"

    def test_get_spending_by_geography_with_recipient_no_agency(self):
        citation = build_tool_citation(
            "get_spending_by_geography",
            {"scope": "place_of_performance", "geo_layer": "county", "start_fiscal_year": 2023, "end_fiscal_year": 2023, "recipient_name": "Boeing"},
        )
        assert "agency_name" not in citation.parameters
        assert citation.description == "Spending by county (place_of_performance), Boeing, FY2023-FY2023"

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
        agency_id=1, agency_name=name, toptier_code="049", abbreviation="NSF", agency_slug="nsf",
        active_fy="2026", active_fq="4", budget_authority_amount=0.0, obligated_amount=0.0,
        outlay_amount=0.0, percentage_of_total_budget_authority=0.0,
        current_total_budget_authority_amount=0.0,
    )


class FakeClient:
    """Stands in for USASpendingClient in _build_filters tests - only
    find_agency_by_name is ever called by _build_filters, so that's all
    that needs faking."""

    def __init__(self, agency: ToptierAgency | None):
        self._agency = agency

    def find_agency_by_name(self, name):
        return self._agency


class TestFormatTopAgenciesByBudget:
    def _agency(self, name, abbreviation, budget_authority_amount, percentage, fy="2026", fq="4"):
        return ToptierAgency(
            agency_id=1, agency_name=name, toptier_code="000", abbreviation=abbreviation,
            agency_slug=name.lower(), active_fy=fy, active_fq=fq,
            budget_authority_amount=budget_authority_amount, obligated_amount=0.0, outlay_amount=0.0,
            percentage_of_total_budget_authority=percentage, current_total_budget_authority_amount=0.0,
        )

    def test_ranked_lines_include_amount_and_percentage(self):
        agencies = [
            self._agency("Department of Health and Human Services", "HHS", 3650342489549.92, 0.2355772266133647),
            self._agency("Department of the Treasury", "TREAS", 3525650703580.85, 0.22753016111083907),
        ]
        result = _format_top_agencies_by_budget(agencies)
        assert "1. Department of Health and Human Services (HHS): $3,650,342,489,549.92 (23.56% of total federal budget authority)" in result
        assert "2. Department of the Treasury (TREAS): $3,525,650,703,580.85 (22.75% of total federal budget authority)" in result

    def test_period_label_reflects_current_fy_fq(self):
        result = _format_top_agencies_by_budget([self._agency("HHS", "HHS", 1.0, 0.1, fy="2026", fq="4")])
        assert result.startswith("As of FY2026 Q4:")

    def test_empty_list(self):
        result = _format_top_agencies_by_budget([])
        assert result == "As of current period:\n"

    def test_never_surfaces_current_total_budget_authority_amount(self):
        # That field is a government-wide total repeated on every agency, not
        # this agency's own figure - must never appear in the formatted output.
        agency = self._agency("HHS", "HHS", 3650342489549.92, 0.2355772266133647)
        result = _format_top_agencies_by_budget([agency])
        assert "15,495,311,418,794" not in result


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

    def test_keywords_becomes_single_item_list(self):
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, keywords="climate research")
        assert filters.keywords == ["climate research"]

    def test_date_type_sets_it_on_the_time_period_object(self):
        # Regression for the real finding: a multi-year award appeared
        # under both FY2023 and FY2024 with the same total under the
        # default action_date matching - new_awards_only eliminates that,
        # verified live 2026-09-06 (see verify_shared_filters.py).
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024, date_type="New Awards Only"
        )
        assert filters.time_period[0].date_type == "new_awards_only"

    def test_omitting_date_type_leaves_it_unset(self):
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024)
        assert filters.time_period[0].date_type is None

    def test_unrecognized_date_type_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized date_type"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, date_type="whenever")

    def test_place_of_performance_and_recipient_scope_are_independent(self):
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024,
            place_of_performance_scope="domestic", recipient_scope="foreign",
        )
        assert filters.place_of_performance_scope == "domestic"
        assert filters.recipient_scope == "foreign"

    def test_unrecognized_scope_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized place_of_performance_scope"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, place_of_performance_scope="martian")

    def test_naics_code_becomes_require_list(self):
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, naics_code="541511")
        assert filters.naics_codes.require == ["541511"]

    def test_malformed_naics_code_raises(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a NAICS code"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, naics_code="software development")

    def test_psc_code_becomes_flat_list_not_hierarchical_object(self):
        # Verified live 2026-09-06: the flat list form (not the
        # require/exclude path object) is what actually filters correctly
        # for a single known code.
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, psc_code="7030")
        assert filters.psc_codes == ["7030"]

    def test_malformed_psc_code_raises(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a PSC code"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, psc_code="software")

    def test_cfda_program_becomes_program_numbers_list(self):
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, cfda_program="10.001")
        assert filters.program_numbers == ["10.001"]

    def test_malformed_cfda_program_raises(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a CFDA"):
            _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024, cfda_program="research grants")

    # agency_name optional / recipient_id (2026-09-08) - real, live-verified
    # findings: recipient_id reproduces a recipient's true all-time total
    # to the penny on get_spending_by_category/get_spending_over_time, but
    # is silently ignored on search_awards - see BACKLOG.md's
    # recipient-profile entry and private/HUMAN_INTERVENTIONS.md #26.

    def test_agency_name_none_with_recipient_name_omits_agencies_filter(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, recipient_name="Boeing")
        assert filters.agencies is None
        assert filters.recipient_search_text == ["Boeing"]

    def test_agency_name_none_with_recipient_id_omits_agencies_filter(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, recipient_id="abc-P")
        assert filters.agencies is None
        assert filters.recipient_id == "abc-P"

    def test_all_scoping_params_none_raises(self):
        with pytest.raises(USASpendingAPIError, match="At least one of"):
            _build_filters(FakeClient(make_agency()), None, 2021, 2024)

    # Place/code/keyword filters are equally legitimate scoping on their
    # own, with no agency or recipient set (#16).

    def test_performed_in_state_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, performed_in_state="Louisiana")
        assert filters.agencies is None
        assert filters.place_of_performance_locations[0].state == "LA"

    def test_recipient_in_state_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, recipient_in_state="Texas")
        assert filters.recipient_locations[0].state == "TX"

    def test_naics_code_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, naics_code="541511")
        assert filters.naics_codes.require == ["541511"]

    def test_psc_code_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, psc_code="7030")
        assert filters.psc_codes == ["7030"]

    def test_cfda_program_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, cfda_program="93.778")
        assert filters.program_numbers == ["93.778"]

    def test_keywords_alone_is_sufficient_scope(self):
        filters = _build_filters(FakeClient(make_agency()), None, 2021, 2024, keywords="climate research")
        assert filters.keywords == ["climate research"]

    def test_award_type_alone_is_not_sufficient_scope(self):
        with pytest.raises(USASpendingAPIError, match="At least one of"):
            _build_filters(FakeClient(make_agency()), None, 2021, 2024, award_type="grants")

    def test_min_amount_alone_is_not_sufficient_scope(self):
        with pytest.raises(USASpendingAPIError, match="At least one of"):
            _build_filters(FakeClient(make_agency()), None, 2021, 2024, min_amount=1_000_000)

    def test_agency_name_given_still_resolves_normally(self):
        # Regression: the common case (agency_name alone) must be
        # unaffected by making it optional.
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024)
        assert filters.agencies[0].name == "National Science Foundation"

    def test_recipient_id_passthrough(self):
        filters = _build_filters(
            FakeClient(make_agency()), "NSF", 2021, 2024, recipient_id="419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"
        )
        assert filters.recipient_id == "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P"

    def test_omitting_recipient_id_leaves_it_unset(self):
        filters = _build_filters(FakeClient(make_agency()), "NSF", 2021, 2024)
        assert filters.recipient_id is None


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


class TestTruncationNote:
    # Regression coverage for a real bug found live via the UI
    # (2026-09-06): a min_amount-filtered search_awards query silently
    # returned 5 of a larger real match set (page_metadata.hasNext was
    # true, but discarded before this fix), and the model presented the
    # partial slice as the complete list of matching awards.

    def test_no_note_when_hasNext_is_false(self):
        assert _truncation_note(False, shown=5) == ""

    def test_note_when_hasNext_is_true(self):
        note = _truncation_note(True, shown=5)
        assert note != ""
        assert "5" in note
        assert "not" in note.lower()  # "not exhaustive"/"not the complete list"

    def test_note_tells_the_model_not_to_claim_completeness(self):
        # The literal wording matters here, not just presence/absence - a
        # vague note ("results may be limited") is easy for a model to
        # skip past; this asserts the instruction is explicit.
        note = _truncation_note(True, shown=5)
        assert "complete" in note.lower() or "exhaustive" in note.lower()


class TestFormatApiMessages:
    # Found live 2026-09-08 via a raw curl to spending_by_award while
    # investigating whether recipient_id silently no-ops there: the API
    # isn't silent, it says exactly what happened ("The following filters
    # from the request were not used: {'recipient_id'}...") in its own
    # `messages` field - which two of three response models already
    # captured and nothing ever read.

    def test_none_messages_produces_empty_string(self):
        assert _format_api_messages(None) == ""

    def test_empty_list_produces_empty_string(self):
        assert _format_api_messages([]) == ""

    def test_single_message_surfaced(self):
        result = _format_api_messages(["The following filters from the request were not used: {'recipient_id'}"])
        assert "recipient_id" in result
        assert "API notice" in result

    def test_multiple_messages_all_surfaced(self):
        result = _format_api_messages(["message one", "message two"])
        assert "message one" in result
        assert "message two" in result


class TestAgencyLabel:
    def test_name_and_abbreviation(self):
        agency = {"toptier_agency": {"name": "National Science Foundation", "abbreviation": "NSF"}}
        assert _agency_label(agency) == "National Science Foundation (NSF)"

    def test_none_agency(self):
        assert _agency_label(None) == "N/A"

    def test_missing_toptier_agency(self):
        assert _agency_label({"toptier_agency": None}) == "N/A"


class TestLocationLabel:
    def test_full_shows_city_and_state(self):
        location = {"city_name": "BOULDER", "state_name": "COLORADO"}
        assert _location_label(location) == "BOULDER, COLORADO"

    def test_full_falls_back_to_state_only(self):
        assert _location_label({"state_name": "VIRGINIA"}) == "VIRGINIA"

    def test_not_full_shows_state_only_even_with_city(self):
        # Used for record_type 1/3 (aggregate/PII-redacted) recipients -
        # found live 2026-09-08 that the API still returns city/county/zip
        # for a redacted individual, which would undercut the redaction if
        # forwarded as-is.
        location = {"city_name": "ZUNI", "state_name": "VIRGINIA", "county_name": "SOUTHAMPTON"}
        assert _location_label(location, full=False) == "VIRGINIA"

    def test_none_location(self):
        assert _location_label(None) == "N/A"


class TestFormatPeriodOfPerformance:
    def test_start_and_end(self):
        pop = {"start_date": "2018-10-01", "end_date": "2028-09-30"}
        assert _format_period_of_performance(pop) == "2018-10-01 to 2028-09-30"

    def test_potential_end_date_shown_when_different(self):
        pop = {"start_date": "2011-12-23", "end_date": "2026-09-30", "potential_end_date": "2026-09-30 00:00:00"}
        result = _format_period_of_performance(pop)
        assert "potential end date" in result

    def test_none_when_both_dates_missing(self):
        # Found live 2026-09-08 on a real SBA guaranteed-loan record: both
        # start_date/end_date were null, and the prior "? to ?" fallback
        # rendered as noise. None (not a placeholder) lets callers skip
        # the line entirely.
        assert _format_period_of_performance({"start_date": None, "end_date": None}) is None

    def test_none_when_pop_missing(self):
        assert _format_period_of_performance(None) is None


class TestFormatContractOrIdv:
    # Trimmed real fixtures from live records pulled 2026-09-08 (LEIDOS
    # NSF contract, IDA NSF IDV) - not synthetic shapes.
    CONTRACT: ClassVar[dict[str, Any]] = {
        "category": "contract",
        "type_description": "DEFINITIVE CONTRACT",
        "piid": "NSFDACS1219442",
        "description": "SCIENCE OPERATION AND MAINTENANCE SUPPORT",
        "total_obligation": 3129062649.79,
        "base_and_all_options": 3174807441.79,
        "date_signed": "2011-12-23",
        "period_of_performance": {"start_date": "2011-12-23", "end_date": "2026-09-30"},
        "awarding_agency": {"toptier_agency": {"name": "National Science Foundation", "abbreviation": "NSF"}},
        "funding_agency": None,
        "recipient": {"recipient_name": "LEIDOS, INC.", "location": {"city_name": "GAITHERSBURG", "state_name": "MARYLAND"}},
        "place_of_performance": {"country_name": "ANTARCTICA"},
        "subaward_count": 1219,
        "total_subaward_amount": 581011041.48,
        "latest_transaction_contract_data": {
            "extent_competed_description": "FULL AND OPEN COMPETITION",
            "number_of_offers_received": "7",
            "type_of_contract_pricing_description": "COST PLUS AWARD FEE",
            "naics_description": "FACILITIES SUPPORT SERVICES",
            "product_or_service_description": "OPERATION OF GOCO R&D FACILITIES",
        },
        "parent_award": None,
    }

    def test_headline_and_money(self):
        result = _format_contract_or_idv(self.CONTRACT)
        assert "DEFINITIVE CONTRACT (NSFDACS1219442)" in result
        assert "$3,129,062,649.79" in result
        assert "$3,174,807,441.79" in result

    def test_recipient_and_place_of_performance(self):
        result = _format_contract_or_idv(self.CONTRACT)
        assert "LEIDOS, INC. (GAITHERSBURG, MARYLAND)" in result
        assert "ANTARCTICA" in result

    def test_competition_detail_fields_included(self):
        result = _format_contract_or_idv(self.CONTRACT)
        assert "FULL AND OPEN COMPETITION" in result
        assert "7 offers received" in result
        assert "FACILITIES SUPPORT SERVICES" in result

    def test_no_parent_award_line_when_none(self):
        assert "parent" not in _format_contract_or_idv(self.CONTRACT).lower()

    def test_parent_award_shown_when_present(self):
        data = {**self.CONTRACT, "parent_award": {
            "piid": "SPE2DX16D1500", "agency_name": "Department of Defense",
            "type_of_idc_description": "INDEFINITE DELIVERY / INDEFINITE QUANTITY",
            "generated_unique_award_id": "CONT_IDV_SPE2DX16D1500_9700",
        }}
        result = _format_contract_or_idv(data)
        assert "parent IDV SPE2DX16D1500" in result

    def test_parent_award_internal_id_shown_for_followup_calls(self):
        # Real gap found live 2026-09-08: a child contract's parent-award
        # line didn't surface the parent IDV's own internal_id, so the
        # model had no way to call get_award_details again on the vehicle
        # itself - it could only ever reach the child contract.
        data = {**self.CONTRACT, "parent_award": {
            "piid": "NSFOIA0408601", "agency_name": "National Science Foundation",
            "type_of_idc_description": "INDEFINITE DELIVERY / INDEFINITE QUANTITY",
            "generated_unique_award_id": "CONT_IDV_NSFOIA0408601_4900",
        }}
        result = _format_contract_or_idv(data)
        assert "internal_id: CONT_IDV_NSFOIA0408601_4900" in result

    def test_no_idv_caveat_for_a_plain_contract(self):
        assert "not itself a spending transaction" not in _format_contract_or_idv(self.CONTRACT)

    def test_idv_caveat_present_for_idv_category(self):
        # Real live finding (2026-09-08): a real NSF IDIQ came back
        # total_obligation=0.0 - the caveat must show for every idv
        # record, not just ones that happen to be zero.
        data = {**self.CONTRACT, "category": "idv", "total_obligation": 0.0}
        result = _format_contract_or_idv(data)
        assert "not itself a spending transaction" in result

    def test_no_period_of_performance_line_when_both_dates_missing(self):
        data = {**self.CONTRACT, "period_of_performance": {"start_date": None, "end_date": None}}
        assert "Period of performance" not in _format_contract_or_idv(data)

    def test_child_order_rollup_replaces_caveat_with_real_numbers(self):
        # Real live finding (2026-09-08): this exact IDV's own
        # total_obligation is $0.00, but it has 237 real child awards
        # totaling $175M - the rollup must show that instead of the
        # generic "doesn't include child-order spending" caveat.
        data = {**self.CONTRACT, "category": "idv", "total_obligation": 0.0}
        rollup = IDVAmountsResponse(
            generated_unique_award_id="CONT_IDV_NSFOIA0408601_4900",
            child_idv_count=0, child_award_count=237,
            child_award_total_obligation=175053251.83,
            child_award_base_and_all_options_value=179258219.93,
            child_award_base_exercised_options_val=176435506.0,
            grandchild_award_count=0, grandchild_award_total_obligation=0.0,
            grandchild_award_base_and_all_options_value=0.0,
            grandchild_award_base_exercised_options_val=0.0,
        )
        result = _format_contract_or_idv(data, child_order_rollup=rollup)
        assert "not itself a spending transaction" not in result
        assert "237 child awards totaling $175,053,251.83" in result

    def test_grandchild_line_suppressed_when_zero(self):
        rollup = IDVAmountsResponse(
            generated_unique_award_id="x", child_idv_count=0, child_award_count=1,
            child_award_total_obligation=1.0, child_award_base_and_all_options_value=1.0,
            child_award_base_exercised_options_val=1.0, grandchild_award_count=0,
            grandchild_award_total_obligation=0.0, grandchild_award_base_and_all_options_value=0.0,
            grandchild_award_base_exercised_options_val=0.0,
        )
        data = {**self.CONTRACT, "category": "idv"}
        assert "grandchild" not in _format_contract_or_idv(data, child_order_rollup=rollup).lower()

    def test_grandchild_line_shown_when_nonzero(self):
        rollup = IDVAmountsResponse(
            generated_unique_award_id="x", child_idv_count=2, child_award_count=25,
            child_award_total_obligation=363410.59, child_award_base_and_all_options_value=297285.59,
            child_award_base_exercised_options_val=297285.59, grandchild_award_count=54,
            grandchild_award_total_obligation=377145.57, grandchild_award_base_and_all_options_value=306964.49,
            grandchild_award_base_exercised_options_val=311020.57,
        )
        data = {**self.CONTRACT, "category": "idv"}
        result = _format_contract_or_idv(data, child_order_rollup=rollup)
        assert "54 grandchild orders" in result
        assert "$377,145.57" in result


class TestFormatGeographyResult:
    def test_includes_per_capita_when_present(self):
        result = GeographyTypeResult(shape_code="CA", display_name="California", aggregated_amount=1024859924.64, population=39538223, per_capita=25.92)
        formatted = _format_geography_result(result)
        assert "California: $1,024,859,924.64" in formatted
        assert "$25.92 per capita" in formatted
        assert "39,538,223" in formatted

    def test_omits_per_capita_when_null(self):
        result = GeographyTypeResult(shape_code="ATA", display_name="Antarctica", aggregated_amount=217092822.46, population=None, per_capita=None)
        formatted = _format_geography_result(result)
        assert formatted == "Antarctica: $217,092,822.46"

    def test_falls_back_to_shape_code_when_display_name_null(self):
        result = GeographyTypeResult(shape_code="1198", display_name=None, aggregated_amount=100.0)
        assert _format_geography_result(result) == "1198: $100.00"

    def test_falls_back_to_placeholder_when_both_null(self):
        result = GeographyTypeResult(shape_code=None, display_name=None, aggregated_amount=100.0)
        assert "Unmapped/unknown location" in _format_geography_result(result)


class TestFormatFinancialAssistance:
    GRANT: ClassVar[dict[str, Any]] = {
        "category": "grant",
        "type_description": "COOPERATIVE AGREEMENT (B)",
        "fain": "1755088",
        "description": "MANAGEMENT AND OPERATION OF NCAR",
        "total_obligation": 1012398088.0,
        "total_funding": 1012398088.0,
        "non_federal_funding": None,
        "record_type": 2,
        "date_signed": "2018-09-19",
        "period_of_performance": {"start_date": "2018-10-01", "end_date": "2028-09-30"},
        "awarding_agency": {"toptier_agency": {"name": "National Science Foundation", "abbreviation": "NSF"}},
        "funding_agency": None,
        "recipient": {
            "recipient_name": "UNIVERSITY CORPORATION FOR ATMOSPHERIC RESEARCH",
            "location": {"city_name": "BOULDER", "state_name": "COLORADO"},
        },
        "place_of_performance": {"city_name": "BOULDER", "state_name": "COLORADO"},
        "subaward_count": 55,
        "total_subaward_amount": 10205739.29,
        "cfda_info": [{"cfda_number": "47.050", "cfda_title": "Geosciences"}],
    }

    def test_grant_shows_total_funding(self):
        result = _format_financial_assistance(self.GRANT)
        assert "Total obligated: $1,012,398,088.00" in result
        assert "Total funding: $1,012,398,088.00" in result

    def test_grant_recipient_shown_normally(self):
        result = _format_financial_assistance(self.GRANT)
        assert "UNIVERSITY CORPORATION FOR ATMOSPHERIC RESEARCH (BOULDER, COLORADO)" in result

    def test_cfda_info_listed(self):
        assert "47.050 Geosciences" in _format_financial_assistance(self.GRANT)

    def test_transaction_obligated_amount_never_shown(self):
        # Deliberately excluded - resolved live (2026-09-08) that this is a
        # File C sum, a separately-timed DATA Act submission from
        # total_obligation's File D2 source, not guaranteed to reconcile.
        data = {**self.GRANT, "transaction_obligated_amount": 975888088.0}
        assert "975,888,088" not in _format_financial_assistance(data)

    def test_account_level_totals_never_shown(self):
        data = {**self.GRANT, "total_account_obligation": 42.0, "total_account_outlay": 42.0}
        assert "42.0" not in _format_financial_assistance(data)

    def test_loan_shows_loan_value_not_total_funding(self):
        loan = {**self.GRANT, "category": "loans", "total_loan_value": 98400000.0, "total_subsidy_cost": 0.0,
                "total_funding": 0.0, "total_obligation": 0.0}
        result = _format_financial_assistance(loan)
        assert "Loan value: $98,400,000.00" in result
        assert "Total funding" not in result

    def test_loan_with_zero_obligation_suppresses_the_line(self):
        # Real live finding (2026-09-08): a $98.4M SBA loan came back
        # total_obligation=0.0 - showing "$0.00" right above the real
        # $98.4M loan value read as contradictory/broken.
        loan = {**self.GRANT, "category": "loans", "total_loan_value": 98400000.0, "total_obligation": 0.0}
        assert "Total obligated" not in _format_financial_assistance(loan)

    def test_loan_with_nonzero_obligation_is_still_shown(self):
        loan = {**self.GRANT, "category": "loans", "total_loan_value": 98400000.0, "total_obligation": 500.0}
        assert "Total obligated: $500.00" in _format_financial_assistance(loan)

    def test_record_type_1_hides_name_and_narrows_location(self):
        # "MULTIPLE RECIPIENTS" (the live sentinel string) must never reach
        # the model verbatim as if it were a real recipient name.
        data = {**self.GRANT, "record_type": 1, "recipient": {
            "recipient_name": "MULTIPLE RECIPIENTS",
            "location": {"city_name": "ZUNI", "state_name": "VIRGINIA"},
        }}
        result = _format_financial_assistance(data)
        assert "MULTIPLE RECIPIENTS" not in result
        assert "Multiple recipients" in result
        assert "aggregate award" in result
        assert "ZUNI" not in result  # narrowed to state-only
        assert "VIRGINIA" in result

    def test_record_type_3_hides_name_and_narrows_location(self):
        data = {**self.GRANT, "record_type": 3, "recipient": {
            "recipient_name": "REDACTED DUE TO PII",
            "location": {"city_name": "ZUNI", "state_name": "VIRGINIA"},
        }}
        result = _format_financial_assistance(data)
        assert "REDACTED DUE TO PII" not in result
        assert "redacted" in result.lower()
        assert "ZUNI" not in result

    def test_record_type_2_shows_real_name_and_full_location(self):
        result = _format_financial_assistance(self.GRANT)
        assert "UNIVERSITY CORPORATION FOR ATMOSPHERIC RESEARCH" in result
        assert "BOULDER" in result


class TestFormatAwardDetailsDispatch:
    def test_contract_category_dispatches_to_contract_formatter(self):
        data = {**TestFormatContractOrIdv.CONTRACT}
        assert "not itself a spending transaction" not in _format_award_details(data)

    def test_idv_category_dispatches_to_contract_formatter_with_caveat(self):
        data = {**TestFormatContractOrIdv.CONTRACT, "category": "idv"}
        assert "not itself a spending transaction" in _format_award_details(data)

    def test_grant_category_dispatches_to_financial_assistance_formatter(self):
        data = {**TestFormatFinancialAssistance.GRANT}
        assert "Assistance Listing" in _format_award_details(data)


class TestScopeLabel:
    def test_agency_name_preferred(self):
        assert _scope_label("NSF", "Leidos", "abc-P") == "NSF"

    def test_falls_back_to_recipient_name(self):
        assert _scope_label(None, "Leidos", None) == "Leidos"

    def test_falls_back_to_recipient_id(self):
        assert _scope_label(None, None, "abc-P") == "abc-P"

    def test_all_none_falls_back_to_placeholder(self):
        # Shouldn't happen in practice - _build_filters guarantees at
        # least one is set - but must not crash if it somehow does.
        assert _scope_label(None, None, None) == "unknown scope"

    def test_falls_back_to_performed_in_state_with_no_agency_or_recipient(self):
        assert _scope_label(None, None, None, performed_in_state="LA") == "LA"

    def test_falls_back_to_cfda_program_with_no_agency_or_recipient(self):
        assert _scope_label(None, None, None, cfda_program="93.778") == "93.778"

    def test_agency_name_still_preferred_over_place_filters(self):
        assert _scope_label("NSF", None, None, performed_in_state="LA") == "NSF"


class TestRecipientFormatting:
    # Fixtures pulled from real live data (Boeing, 2026-09-08) - not
    # synthetic shapes. Real live values confirmed: 546 total matches for
    # "Leidos" alone resolve to 6+ distinct recipient_ids sharing the
    # identical name "LEIDOS, INC.", and the parent-level rollup
    # reproduces a recipient's true all-time total to the penny.

    BOEING_LISTING = RecipientListing(
        name="THE BOEING COMPANY", duns="009256819", uei="NU2UC8MX6NK1",
        id="419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", amount=30309729588.71, recipient_level="P",
    )

    BOEING_LOCATION = RecipientLocation(
        address_line1="929 LONG BRIDGE DR", city_name="ARLINGTON", state_code="VA",
        zip="22202", country_name="UNITED STATES",
    )

    BOEING_OVERVIEW = RecipientOverview(
        name="THE BOEING COMPANY", alternate_names=["BOEING CO"], duns="009256819", uei="NU2UC8MX6NK1",
        recipient_id="419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P", recipient_level="P",
        parent_id="419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P",  # self - real live shape, not a real parent
        location=BOEING_LOCATION,
        business_types=["corporate_entity_not_tax_exempt", "us_owned_business"],
        total_transaction_amount=439084269687.36, total_transactions=346764,
        total_face_value_loan_amount=0.0, total_face_value_loan_transactions=0,
    )

    REDACTED_OVERVIEW = RecipientOverview(
        name="REDACTED DUE TO PII", alternate_names=[], duns=None, uei=None,
        recipient_id="6e4362a8-7dd7-8d86-d2ff-8faa5eefe0aa-R", recipient_level="R",
        parent_id=None, location=RecipientLocation(city_name="PLANT CITY", state_code="FL"),
        business_types=[],
        total_transaction_amount=161426671098.53, total_transactions=29994629,
        total_face_value_loan_amount=168657558594.56, total_face_value_loan_transactions=2188551,
    )

    def test_format_recipient_level(self):
        assert _format_recipient_level("P") == "parent"
        assert _format_recipient_level("C") == "child"
        assert _format_recipient_level("R") == "standalone"

    def test_format_business_type_known_codes(self):
        assert _format_business_type("category_business") == "Business"
        assert _format_business_type("us_owned_business") == "U.S. Owned Business"
        assert _format_business_type("sba_certified_8a_joint_venture") == "SBA Certified 8a Joint Venture"

    def test_format_business_type_unknown_code_falls_back_to_title_case(self):
        assert _format_business_type("some_new_flag") == "Some New Flag"

    def test_format_recipient_listing_includes_amount_labeled_as_last_12_months(self):
        result = _format_recipient_listing(self.BOEING_LISTING)
        assert "$30,309,729,588.71" in result
        assert "last 12 months" in result

    def test_format_recipient_listing_includes_recipient_id_for_followup(self):
        result = _format_recipient_listing(self.BOEING_LISTING)
        assert "recipient_id: 419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P" in result

    def test_format_recipient_listing_shows_level(self):
        assert "[parent]" in _format_recipient_listing(self.BOEING_LISTING)

    def test_format_recipient_address_full_for_a_normal_business(self):
        # Confirmed against the real live usaspending.gov Boeing page,
        # pasted in by hand: full street address is shown for a business.
        result = _format_recipient_address(self.BOEING_LOCATION)
        assert "929 LONG BRIDGE DR" in result
        assert "ARLINGTON" in result
        assert "VA" in result

    def test_format_recipient_state_only_hides_street_address(self):
        # Used for the redacted/aggregate bucket case - narrower than
        # _format_recipient_address deliberately, not an omission.
        result = _format_recipient_state_only(self.BOEING_LOCATION)
        assert "929 LONG BRIDGE DR" not in result
        assert result == "VA"

    def test_format_recipient_overview_normal_shows_real_name_and_full_address(self):
        result = _format_recipient_overview(self.BOEING_OVERVIEW)
        assert "THE BOEING COMPANY" in result
        assert "929 LONG BRIDGE DR" in result
        assert "$439,084,269,687.36" in result

    def test_format_recipient_overview_business_types_reformatted(self):
        result = _format_recipient_overview(self.BOEING_OVERVIEW)
        assert "Corporate Entity Not Tax Exempt" in result
        assert "U.S. Owned Business" in result
        assert "corporate_entity_not_tax_exempt" not in result
        assert "Us Owned Business" not in result

    def test_format_recipient_overview_loan_fields_shown_even_at_zero(self):
        # Deliberate: unlike get_award_details's loan case, the live
        # usaspending.gov page itself always shows this line ("$0 from 0
        # transactions") - confirmed by pasting the real page in - so it's
        # never suppressed here.
        result = _format_recipient_overview(self.BOEING_OVERVIEW)
        assert "Face value of loans: $0.00" in result

    def test_format_recipient_overview_self_referential_parent_not_shown_as_a_separate_entity(self):
        # Real live shape: a P-level recipient's own parent_id can equal
        # its own recipient_id (not a real distinct parent) - must not
        # print "Parent: THE BOEING COMPANY [recipient_id: <itself>]".
        result = _format_recipient_overview(self.BOEING_OVERVIEW)
        assert "Parent:" not in result

    def test_format_recipient_overview_real_parent_shown_with_id(self):
        data = self.BOEING_OVERVIEW.model_copy(
            update={"parent_id": "different-id-P", "parent_name": "SOME PARENT CORP"}
        )
        result = _format_recipient_overview(data)
        assert "Parent: SOME PARENT CORP [recipient_id: different-id-P]" in result

    def test_format_recipient_overview_redacted_bucket_hides_sentinel_name(self):
        result = _format_recipient_overview(self.REDACTED_OVERVIEW)
        assert "REDACTED DUE TO PII" not in result.split("\n")[0]  # not presented as a real name
        assert "pooled aggregate" in result

    def test_format_recipient_overview_redacted_bucket_narrows_location(self):
        result = _format_recipient_overview(self.REDACTED_OVERVIEW)
        assert "PLANT CITY" not in result
        assert "FL" in result

    def test_format_recipient_overview_redacted_bucket_still_shows_totals(self):
        # The totals are real (they're just not one entity's totals) - not
        # hidden, just caveated. Formatting must not crash or omit them.
        result = _format_recipient_overview(self.REDACTED_OVERVIEW)
        assert "$161,426,671,098.53" in result


class TestNormalizeDateType:
    def test_case_and_spacing_insensitive(self):
        assert _normalize_date_type("New Awards Only") == "new_awards_only"
        assert _normalize_date_type("new-awards-only") == "new_awards_only"
        assert _normalize_date_type("  action_date  ") == "action_date"

    def test_all_four_real_api_values_accepted(self):
        # action_date, date_signed, last_modified_date, new_awards_only -
        # per search_filters.md's Award Search Time Period Object.
        assert VALID_DATE_TYPES == {"action_date", "date_signed", "last_modified_date", "new_awards_only"}
        for value in VALID_DATE_TYPES:
            assert _normalize_date_type(value) == value

    def test_garbage_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized date_type 'whenever'"):
            _normalize_date_type("whenever")


class TestNormalizeScope:
    def test_domestic_and_foreign_accepted(self):
        assert _normalize_scope("domestic", "recipient_scope") == "domestic"
        assert _normalize_scope("Foreign", "recipient_scope") == "foreign"

    def test_garbage_raises_with_the_right_param_name(self):
        with pytest.raises(USASpendingAPIError, match="Unrecognized recipient_scope 'martian'"):
            _normalize_scope("martian", "recipient_scope")


class TestCodeValidation:
    # Format-only validation, not existence checks against a real
    # vocabulary - NAICS/PSC/CFDA are large government classification
    # systems with no small closed list to hardcode, unlike award_type/
    # state. See ADVANCED_FILTER_FIELD_COVERAGE's naics_codes entry.

    def test_valid_naics_codes_of_various_lengths(self):
        assert _validate_naics_code("54") == "54"
        assert _validate_naics_code("541511") == "541511"

    def test_naics_code_rejects_non_numeric(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a NAICS code"):
            _validate_naics_code("information technology")

    def test_valid_psc_code(self):
        assert _validate_psc_code("7030") == "7030"

    def test_psc_code_rejects_wrong_length(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a PSC code"):
            _validate_psc_code("70")

    def test_valid_cfda_program(self):
        assert _validate_cfda_program("10.001") == "10.001"

    def test_cfda_program_rejects_wrong_format(self):
        with pytest.raises(USASpendingAPIError, match="doesn't look like a CFDA"):
            _validate_cfda_program("10-001")


class TestNormalizeCategory:
    # Regression coverage for the real gap: category was a bare str with
    # the valid values only in docstring prose (same anti-pattern
    # AWARD_TYPE_GROUPS already fixed for award_type) - an unrecognized
    # category previously reached the live API and came back as a raw
    # 404 with no guidance; now it's a clean, code-owned error before any
    # network call happens.

    def test_all_fifteen_live_verified_categories_accepted(self):
        # Verified live 2026-09-07 that every one of these actually works
        # against the real API for a real agency/fiscal-year query -
        # dev_tools verification, not just this list matching itself.
        expected = {
            "awarding_agency", "awarding_subagency", "cfda", "country", "county",
            "defc", "district", "federal_account", "funding_agency",
            "funding_subagency", "naics", "psc", "recipient", "recipient_duns",
            "state_territory",
        }
        assert VALID_CATEGORIES == expected
        for category in expected:
            assert _normalize_category(category) == category

    def test_case_and_spacing_insensitive(self):
        assert _normalize_category("NAICS") == "naics"
        assert _normalize_category("  State Territory  ") == "state_territory"
        assert _normalize_category("awarding-agency") == "awarding_agency"

    def test_documented_but_404ing_categories_are_rejected(self):
        # These ARE in the live API's own top-level contract enum but 404
        # in practice (BACKLOG.md's "Daily health check" entry) - the
        # whole point of validating in code is catching these before a
        # network round trip, not just catching made-up category names.
        for category in ("object_class", "program_activity", "recipient_parent_duns", "tas"):
            with pytest.raises(USASpendingAPIError, match="Unknown category"):
                _normalize_category(category)

    def test_garbage_raises_with_the_full_valid_list(self):
        with pytest.raises(USASpendingAPIError, match="Unknown category 'vendor'"):
            _normalize_category("vendor")


class TestNormalizeGroup:
    # Regression coverage: group previously had NO code-side validation at
    # all (BACKLOG.md judged this "safe in practice" since the live API's
    # own error for a bad value is already clean) - added for consistency
    # once every other fixed-vocabulary param got both a schema-level
    # Literal and a runtime check.

    def test_all_four_real_api_values_accepted(self):
        assert VALID_GROUPS == {"fiscal_year", "calendar_year", "quarter", "month"}
        for group in VALID_GROUPS:
            assert _normalize_group(group) == group

    def test_case_and_spacing_insensitive(self):
        assert _normalize_group("Fiscal Year") == "fiscal_year"
        assert _normalize_group("CALENDAR-YEAR") == "calendar_year"

    def test_garbage_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unknown group 'week'"):
            _normalize_group("week")


class TestNormalizeRecipientAwardType:
    # POST /api/v2/recipient/'s own award_type enum (recipient.md) - a
    # real, different, coarser vocabulary from AWARD_TYPE_GROUPS: 6 broad
    # buckets, no sub-type granularity (no cooperative_agreement, no
    # bpa_call), confirmed from the live contract.

    def test_all_six_real_values_accepted(self):
        assert RECIPIENT_AWARD_TYPES == {
            "all", "contracts", "grants", "loans", "direct_payments", "other_financial_assistance",
        }
        for award_type in RECIPIENT_AWARD_TYPES:
            assert _normalize_recipient_award_type(award_type) == award_type

    def test_case_and_spacing_insensitive(self):
        assert _normalize_recipient_award_type("Direct Payments") == "direct_payments"

    def test_garbage_raises(self):
        with pytest.raises(USASpendingAPIError, match="Unknown award_type"):
            _normalize_recipient_award_type("cooperative_agreement")


class TestLiteralTypesMatchVocabulary:
    # The whole point of adding typing.Literal types (so beta_tool's schema
    # generation emits a real JSON-schema enum, constraining what the model
    # can generate at the tool-call level, not just validating it after the
    # fact) only holds if the Literal stays in sync with the runtime
    # vocabulary it's supposed to mirror. These were written as static
    # Literals, not derived from the dicts/sets, specifically to avoid any
    # uncertainty around dynamic Literal construction interacting with
    # `from __future__ import annotations` - which means nothing stops them
    # from silently drifting apart by hand-edit. This is what actually
    # keeps them honest instead.

    def test_award_type_literal_matches_award_type_groups(self):
        assert set(get_args(AwardType)) == set(AWARD_TYPE_GROUPS)

    def test_category_literal_matches_valid_categories(self):
        assert set(get_args(Category)) == VALID_CATEGORIES

    def test_date_type_literal_matches_valid_date_types(self):
        assert set(get_args(DateType)) == VALID_DATE_TYPES

    def test_group_literal_matches_valid_groups(self):
        assert set(get_args(Group)) == VALID_GROUPS

    def test_scope_literal_is_domestic_foreign(self):
        assert set(get_args(Scope)) == {"domestic", "foreign"}

    def test_recipient_award_type_literal_matches_recipient_award_types(self):
        assert set(get_args(RecipientAwardType)) == RECIPIENT_AWARD_TYPES


class TestClampLimit:
    # Regression coverage for a real finding (BACKLOG.md "Red team:
    # resource abuse"): the model passed limit=1000 unprompted and got a
    # raw 422 from the live API ("above max 100"). Clamping silently is
    # safe because page_metadata.hasNext + _truncation_note already tell
    # the model honestly when a clamped set isn't exhaustive.

    def test_within_range_passes_through_unchanged(self):
        assert _clamp_limit(5) == 5
        assert _clamp_limit(100) == 100

    def test_above_max_clamps_to_max(self):
        assert _clamp_limit(1000) == MAX_LIMIT
        assert _clamp_limit(101) == MAX_LIMIT

    def test_zero_or_negative_floors_to_one(self):
        assert _clamp_limit(0) == 1
        assert _clamp_limit(-5) == 1


class TestToolCallBudget:
    # Regression coverage for a real finding (BACKLOG.md "Red team:
    # resource abuse"): "look up the toptier code for NSF, NASA, EPA, DOE,
    # and DOD" triggered 5 real, uncapped API calls in one turn, with
    # nothing stopping a much longer list.

    def test_returns_none_when_no_log_is_active(self):
        # Mirrors _record_tool_call's own "no-op outside ask()" behavior -
        # a tool called directly (tests, dev_tools scripts) shouldn't be
        # budget-gated at all, only real agent turns.
        token = _tool_call_log.set(None)
        try:
            assert _check_tool_call_budget() is None
        finally:
            _tool_call_log.reset(token)

    def test_returns_none_when_under_the_cap(self):
        token = _tool_call_log.set([("lookup_agency", None, {})] * (MAX_TOOL_CALLS_PER_TURN - 1))
        try:
            assert _check_tool_call_budget() is None
        finally:
            _tool_call_log.reset(token)

    def test_returns_an_error_string_at_the_cap(self):
        token = _tool_call_log.set([("lookup_agency", None, {})] * MAX_TOOL_CALLS_PER_TURN)
        try:
            result = _check_tool_call_budget()
            assert result is not None
            assert str(MAX_TOOL_CALLS_PER_TURN) in result
        finally:
            _tool_call_log.reset(token)


class TestExtractGuideQuestion:
    # Regression coverage: replacing page-number citations with the real
    # question, per real chunk data - verified live against
    # data/chunks/analysts_guide_chunks.jsonl that only 42/70 (60%) of
    # Guide chunks are actually Q&A-shaped, not all of them as might be
    # assumed from the chunking strategy's name.

    def test_extracts_a_real_smart_quoted_question(self):
        text = "‘What is an obligation?’\nAn obligation is a promise made by the government..."
        assert _extract_guide_question(text) == "What is an obligation?"

    def test_returns_none_for_a_table_of_contents_chunk(self):
        text = "Analyst’s Guide to Federal Spending Data\nContents\nAWARD SPENDING ....... 3"
        assert _extract_guide_question(text) is None

    def test_returns_none_for_a_bare_section_header_chunk(self):
        assert _extract_guide_question("AWARD AND ACCOUNT SPENDING COMPARISON") is None

    def test_only_checks_the_first_line(self):
        # A question mark appearing later in the chunk (inside the answer
        # text, say) shouldn't make an otherwise non-Q&A chunk match.
        text = "AWARD SPENDING\nIs this a question? No, just a header chunk with a '?' in the body."
        assert _extract_guide_question(text) is None

    def test_extracts_an_unquoted_question(self):
        # Real bug found live (2026-09-07): some questions have no quote
        # at all in the source PDF (e.g. "What is a recipient?" right
        # after a "RECIPIENT DATA ELEMENTS" section header) - confirmed
        # directly against the raw PDF text. ingest.py's chunker was
        # fixed to split on these; this is the matching extraction-side
        # fix so the resulting unquoted chunk is still recognized.
        text = "What is a recipient?\nA recipient is a company, organization..."
        assert _extract_guide_question(text) == "What is a recipient?"

    def test_extracts_a_question_with_an_opening_quote_but_no_closing_one(self):
        # Real bug found live: the source PDF sometimes drops the closing
        # quote glyph entirely rather than mis-rendering it - confirmed
        # directly against the raw PDF text for "What are the two major
        # categories of award spending?" (opening quote present, no
        # closing quote anywhere on the line). A strict quote-*pair*
        # regex missed this even though the chunk itself was already
        # correctly split - fixed by stripping quotes independently.
        text = "‘What are the two major categories of award spending?\nThe two main categories are..."
        assert _extract_guide_question(text) == "What are the two major categories of award spending?"

    def test_extracts_a_question_with_a_closing_quote_but_no_opening_one(self):
        # Mirror image, also found live: a closing quote with no opening
        # one at all, e.g. "Where is the full list of agency names and
        # codes?'" - confirmed directly against the raw PDF text.
        text = "Where is the full list of agency names and codes?’\nThe agency_codes.csv file..."
        assert _extract_guide_question(text) == "Where is the full list of agency names and codes?"


class TestBuildGuideCitation:
    def test_glossary_chunk_with_a_slug_gets_a_per_term_url(self):
        # Verified live (2026-09-07): https://www.usaspending.gov/?glossary=
        # <slug> opens the site with the glossary sidebar pre-opened to that
        # exact term - a genuine per-term deep link, unlike the Guide's one
        # fixed URL for every citation.
        chunk = {
            "id": "Glossary_obligation",
            "source": "USASpending Glossary",
            "term": "Obligation",
            "slug": "obligation",
            "text": "...",
        }
        citation = _build_guide_citation(chunk)
        assert citation == Citation(
            chunk_id="Glossary_obligation",
            source="USASpending Glossary",
            term="Obligation",
            url=f"{GLOSSARY_URL_BASE}obligation",
        )

    def test_glossary_chunk_with_no_slug_gets_a_term_citation_no_url(self):
        # Defensive fallback - every real Glossary chunk carries a slug
        # (ingest_glossary.py always sets it), but don't fabricate a broken
        # link if one is ever missing.
        chunk = {"id": "glossary_1", "source": "USASpending Glossary", "term": "Obligation", "text": "..."}
        citation = _build_guide_citation(chunk)
        assert citation == Citation(chunk_id="glossary_1", source="USASpending Glossary", term="Obligation")
        assert citation.url is None

    def test_qa_shaped_guide_chunk_gets_a_question_and_the_live_url(self):
        chunk = {
            "id": "Analyst's_Guide_p5",
            "source": "Analyst's Guide",
            "text": "‘What is an obligation?’\nAn obligation is...",
            "page_start": 5,
        }
        citation = _build_guide_citation(chunk)
        assert citation.question == "What is an obligation?"
        assert citation.url == GUIDE_URL
        assert citation.page is None

    def test_non_qa_guide_chunk_falls_back_to_page_but_still_gets_the_url(self):
        # The url is unconditional for every Guide citation - it isn't
        # per-question anchor-addressable (verified live: the real page
        # is a client-rendered SPA with no server-side anchors at all),
        # so linking it even for the page-fallback case is still correct,
        # just less specific.
        chunk = {
            "id": "Analyst's_Guide_p1",
            "source": "Analyst's Guide",
            "text": "AWARD AND ACCOUNT SPENDING COMPARISON",
            "page_start": 2,
        }
        citation = _build_guide_citation(chunk)
        assert citation.question is None
        assert citation.page == 2
        assert citation.url == GUIDE_URL


class TestGuideQuestionDedup:
    # The user's specific ask: dedup citations by the actual question
    # text, not just chunk_id - a single logical Q&A entry can span more
    # than one physical chunk (long answers get split further beyond the
    # Q&A boundary), which would otherwise show the same question twice
    # under two different chunk_ids.

    def test_two_different_chunk_ids_same_question_dedups_to_one_citation(self):
        chunks = [
            {
                "id": "Analyst's_Guide_p5",
                "source": "Analyst's Guide",
                "text": "‘What is an obligation?’\nFirst half of a long answer...",
                "page_start": 5,
            },
            {
                "id": "Analyst's_Guide_p6",
                "source": "Analyst's Guide",
                "text": "‘What is an obligation?’\nSecond half of the same long answer, chunked separately...",
                "page_start": 6,
            },
        ]
        # Mirrors ask()'s own dedup loop in orchestrator.py, tested at the
        # unit level via the same building blocks it uses.
        seen_chunk_ids: set[str] = set()
        seen_questions: set[str] = set()
        citations = []
        for chunk in chunks:
            if chunk["id"] in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk["id"])
            citation = _build_guide_citation(chunk)
            if citation.question is not None:
                normalized = citation.question.strip().lower()
                if normalized in seen_questions:
                    continue
                seen_questions.add(normalized)
            citations.append(citation)

        assert len(citations) == 1
        assert citations[0].chunk_id == "Analyst's_Guide_p5"

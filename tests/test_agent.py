from datetime import date, datetime, timezone
from typing import get_args

import pytest

from backend.app.agent.response_shaping import (
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
    US_STATE_ABBREVIATIONS,
    VALID_DATE_TYPES,
    AwardType,
    DateType,
    Scope,
    _amount_field_for_award_type,
    _build_filters,
    _clamp_limit,
    _normalize_award_type,
    _normalize_date_type,
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
    _check_tool_call_budget,
    _normalize_category,
    _normalize_group,
    _tool_call_log,
    _truncation_note,
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

    def test_get_agency_budget_never_charts(self):
        assert should_chart("get_agency_budget", make_category_response(5)) is None

    def test_unknown_tool_name_returns_none(self):
        assert should_chart("some_future_tool", make_category_response(5)) is None


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


class TestBuildGuideCitation:
    def test_glossary_chunk_gets_a_term_citation_no_url(self):
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

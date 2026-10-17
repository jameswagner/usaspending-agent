from unittest.mock import MagicMock, patch

from backend.app.agent.download_handler import (
    DownloadIntent,
    _award_id_endpoint,
    handle_download_followup,
)
from backend.app.usaspending import (
    DownloadJobResponse,
    DownloadStatusResponse,
    ToptierAgency,
    USASpendingAPIError,
)


def make_agency(name="National Science Foundation"):
    return ToptierAgency(
        agency_id=1, agency_name=name, toptier_code="4900", abbreviation="NSF",
        agency_slug="nsf", active_fy="2026", active_fq="4",
        budget_authority_amount=1.0, obligated_amount=1.0, outlay_amount=1.0,
        percentage_of_total_budget_authority=0.01, current_total_budget_authority_amount=1.0,
    )


class FakeDownloadClient:
    """Stands in for USASpendingClient - only what the download pipeline calls is faked."""

    def __init__(self, agency=None, job=None, statuses=None, raises=None):
        self._agency = agency
        self._job = job
        self._statuses = list(statuses or [])
        self._raises = raises
        self.last_filters = None
        self.last_columns = None
        self.last_spending_level = None

    def find_agency_by_name(self, name):
        return self._agency

    def find_agency_by_name_with_subagency(self, name):
        return (self._agency, None) if self._agency is not None else None

    def download_search(self, filters, columns, spending_level, file_format="csv"):
        self.last_filters = filters
        self.last_columns = columns
        self.last_spending_level = spending_level
        if self._raises:
            raise self._raises
        return self._job

    def get_download_status(self, file_name):
        if self._raises:
            raise self._raises
        next_item = self._statuses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item

    def _single_award_download(self, award_id):
        self.last_award_id = award_id
        if self._raises:
            raise self._raises
        return self._job

    def download_contract(self, award_id, file_format="csv"):
        return self._single_award_download(award_id)

    def download_assistance(self, award_id, file_format="csv"):
        return self._single_award_download(award_id)

    def download_idv(self, award_id, file_format="csv"):
        return self._single_award_download(award_id)


def make_tool_use_response(input_dict):
    block = MagicMock(type="tool_use", input=input_dict)
    return MagicMock(content=[block])


class TestHandleDownloadFollowup:


    def test_unresolvable_agency_returns_message_without_downloading(self):
        intent = DownloadIntent(agency_raw="Not A Real Agency", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=None)
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert "No agency found matching 'Not A Real Agency'" in result.answer_text
        assert result.downloads == []
        assert client.last_filters is None

    def test_finished_on_first_poll_returns_ready_download(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=456,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result is not None
        assert len(result.downloads) == 1
        assert result.downloads[0].status == "finished"
        assert result.downloads[0].url == finished.file_url
        assert result.downloads[0].total_rows == 456
        assert len(result.tool_citations) == 1
        assert result.tool_citations[0].tool_name == "download_search"
        assert result.tool_citations[0].parameters["agency_name"] == "National Science Foundation"
        assert result.tool_citations[0].parameters["spending_level"] == "awards"
        # Compatibility table: "awards" preserves the legacy /download/awards/ bundling.
        assert client.last_spending_level == ["awards", "subawards"]

    def test_month_level_intent_scopes_filter_to_that_month_not_the_whole_year(self):
        # Regression (live-reported): month-less extraction silently expanded "January 2024" to all of 2024.
        intent = DownloadIntent(
            agency_raw="NSF", time_period_type="calendar", start_date="2024-01-01", end_date="2024-01-31",
        )
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_followup(intent.model_dump(), "conv-1")
        assert client.last_filters.time_period[0].start_date == "2024-01-01"
        assert client.last_filters.time_period[0].end_date == "2024-01-31"

    def test_different_months_in_the_same_year_produce_different_filters(self):
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )

        january = DownloadIntent(agency_raw="NSF", start_date="2024-01-01", end_date="2024-01-31")
        client_jan = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client_jan):
            handle_download_followup(january.model_dump(), "conv-1")

        february = DownloadIntent(agency_raw="NSF", start_date="2024-02-01", end_date="2024-02-29")
        client_feb = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client_feb):
            handle_download_followup(february.model_dump(), "conv-1")

        assert client_jan.last_filters.time_period != client_feb.last_filters.time_period

    def test_poll_timeout_returns_still_running_download(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        running = DownloadStatusResponse(
            status="running", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        # Enough "running" statuses to exhaust the poll loop regardless of interval/timeout constants.
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[running] * 50)
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result is not None
        assert result.downloads[0].status == "running"
        assert result.downloads[0].url == running.file_url
        assert result.downloads[0].status_url == job.status_url

    def test_ready_status_keeps_polling_instead_of_stopping_immediately(self):
        # Regression: a freshly queued job answers "ready" first, confirmed live.
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        ready = DownloadStatusResponse(
            status="ready", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=456,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[ready, finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result.downloads[0].status == "finished"
        assert result.downloads[0].url == finished.file_url

    def test_early_404_on_status_check_is_tolerated_and_retried(self):
        # Regression: an immediate status check can 404 before the job record is indexed, confirmed live.
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=456,
        )
        early_404 = USASpendingAPIError("404: Download job with filename x.zip does not exist.")
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[early_404, finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result.downloads[0].status == "finished"

    def test_failed_status_returns_failure_message(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        failed = DownloadStatusResponse(
            status="failed", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", message="internal error",
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[failed])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result is not None
        assert "failed to generate" in result.answer_text
        assert "internal error" in result.answer_text
        assert result.downloads[0].status == "failed"

    def test_api_error_returns_error_message(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=make_agency(), raises=USASpendingAPIError("500: boom"))
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        assert result is not None
        assert "This download failed" in result.answer_text
        assert result.downloads == []


class TestSpendingLevel:
    """Compatibility-table mapping from DownloadIntent.spending_level to the API's array -
    see #277: /download/search/'s spending_level members are fully independent (["awards"]
    alone excludes sub-awards), unlike the legacy /download/awards/ and /download/transactions/
    endpoints this module preserves parity with."""

    def _run(self, spending_level, agency="NSF"):
        intent = DownloadIntent(agency_raw=agency, start_year=2024, end_year=2024, spending_level=spending_level)
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(agency), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_followup(intent.model_dump(), "conv-1")
        return result, client

    def test_awards_maps_to_awards_and_subawards(self):
        _, client = self._run("awards")
        assert client.last_spending_level == ["awards", "subawards"]

    def test_transactions_maps_to_transactions_and_subawards(self):
        _, client = self._run("transactions")
        assert client.last_spending_level == ["transactions", "subawards"]

    def test_subawards_maps_to_subawards_only(self):
        # Net-new - no legacy single-purpose endpoint to preserve parity with here.
        _, client = self._run("subawards")
        assert client.last_spending_level == ["subawards"]

    def test_transactions_uses_transaction_shaped_columns_not_award_shaped(self):
        _, client = self._run("transactions")
        assert "federal_action_obligation" in client.last_columns
        assert "total_obligated_amount" not in client.last_columns

    def test_subawards_uses_subaward_shaped_columns_not_award_shaped(self):
        # Regression risk: award-level column names (award_id_piid, total_obligated_amount)
        # are silently accepted by the API at request time for this level but make the job
        # fail async - confirmed live 2026-09-24. Must use prime_award_*/subaward* names.
        _, client = self._run("subawards")
        assert "subaward_amount" in client.last_columns
        assert "award_id_piid" not in client.last_columns

    def test_citation_records_resolved_spending_level(self):
        result, _ = self._run("transactions")
        assert result.tool_citations[0].parameters["spending_level"] == "transactions"


class TestDownloadIdentifierColumns:
    """award_id_piid/award_id_fain selection, by award_type - see
    _identifier_columns_for_award_type's own docstring. Live-verified 2026-10-01:
    requesting the identifier column that's 100% null for a narrowed award_type
    (award_id_fain for a contracts-only download, or award_id_piid for a
    grants-only one) crashes /api/v2/download/search/ outright with a generic
    "An error occurred." - not merely a wasted column, a fatal one."""

    def _run(self, award_type, spending_level="awards"):
        intent = DownloadIntent(
            agency_raw="NSF", start_year=2024, end_year=2024,
            award_type=award_type, spending_level=spending_level,
        )
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_followup(intent.model_dump(), "conv-1")
        return client.last_columns

    def test_unscoped_award_type_requests_both_identifiers(self):
        columns = self._run(award_type=None)
        assert "award_id_piid" in columns
        assert "award_id_fain" in columns

    def test_contracts_requests_piid_only(self):
        columns = self._run(award_type="contracts")
        assert "award_id_piid" in columns
        assert "award_id_fain" not in columns

    def test_idv_requests_piid_only(self):
        columns = self._run(award_type="idv")
        assert "award_id_piid" in columns
        assert "award_id_fain" not in columns

    def test_grants_requests_fain_only(self):
        columns = self._run(award_type="grants")
        assert "award_id_fain" in columns
        assert "award_id_piid" not in columns

    def test_cooperative_agreement_sub_type_requests_fain_only(self):
        """cooperative_agreement is a grants sub-type (code 05), not its own broad
        category - must resolve the same way "grants" itself does, not fall through
        to a default."""
        columns = self._run(award_type="cooperative_agreement")
        assert "award_id_fain" in columns
        assert "award_id_piid" not in columns

    def test_loans_requests_fain_only(self):
        columns = self._run(award_type="loans")
        assert "award_id_fain" in columns
        assert "award_id_piid" not in columns

    def test_subawards_level_unaffected_by_award_type(self):
        """subawards has its own fixed prime_award_piid column, never swapped -
        the piid/fain split only applies to "awards"/"transactions"."""
        columns = self._run(award_type="grants", spending_level="subawards")
        assert columns == [
            "prime_award_piid", "subawardee_name", "subaward_amount",
            "subaward_action_date", "prime_award_awarding_agency_name",
        ]


class TestDownloadCategoryColumns:
    """naics_code/product_or_service_code columns, added only when the resolved
    intent was actually scoped by that filter - see _category_columns_for's own
    docstring. Live-verified 2026-10-01: the *search* endpoint's own display field
    names ("NAICS", "PSC") are NOT valid download columns and crash the job the
    same "accepted at request time, fails async" way an always-null identifier
    column does."""

    def _run(self, naics_code=None, psc_code=None, category=None, spending_level="awards"):
        intent = DownloadIntent(
            agency_raw="NSF", start_year=2024, end_year=2024,
            naics_code=naics_code, psc_code=psc_code, category=category, spending_level=spending_level,
        )
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_followup(intent.model_dump(), "conv-1")
        return client.last_columns

    def test_no_category_filter_adds_no_category_column(self):
        columns = self._run()
        assert "naics_code" not in columns
        assert "product_or_service_code" not in columns

    def test_naics_scoped_adds_naics_code_column(self):
        columns = self._run(naics_code="518210")
        assert "naics_code" in columns
        # The search endpoint's display field name must never leak in here - it
        # crashes the download job (see class docstring).
        assert "NAICS" not in columns

    def test_psc_scoped_adds_product_or_service_code_column(self):
        columns = self._run(psc_code="7030")
        assert "product_or_service_code" in columns
        assert "PSC" not in columns
        assert "psc_code" not in columns

    def test_both_naics_and_psc_scoped_adds_both_columns(self):
        columns = self._run(naics_code="518210", psc_code="7030")
        assert "naics_code" in columns
        assert "product_or_service_code" in columns

    def test_subawards_level_unaffected_by_category_filters(self):
        columns = self._run(naics_code="518210", spending_level="subawards")
        assert "naics_code" not in columns

    def test_naics_breakdown_adds_naics_code_column_with_no_filter(self):
        """The original repro: a NAICS *breakdown* (ranking across every code, not
        filtered to one) has no naics_code scope value, but each row still has its
        own real NAICS code worth showing - live-verified 2026-10-01 that naics_code
        populates fine as a download column with zero naics_codes filter applied."""
        columns = self._run(category="naics")
        assert "naics_code" in columns

    def test_psc_breakdown_adds_product_or_service_code_column(self):
        columns = self._run(category="psc")
        assert "product_or_service_code" in columns

    def test_cfda_breakdown_adds_cfda_number_column(self):
        columns = self._run(category="cfda")
        assert "cfda_number" in columns

    def test_country_breakdown_adds_pop_country_code_column(self):
        columns = self._run(category="country")
        assert "primary_place_of_performance_country_code" in columns

    def test_county_breakdown_adds_pop_county_name_column(self):
        columns = self._run(category="county")
        assert "primary_place_of_performance_county_name" in columns

    def test_state_territory_breakdown_adds_pop_state_code_column(self):
        columns = self._run(category="state_territory")
        assert "primary_place_of_performance_state_code" in columns

    def test_awarding_subagency_breakdown_adds_column(self):
        columns = self._run(category="awarding_subagency")
        assert "awarding_sub_agency_name" in columns

    def test_funding_agency_breakdown_adds_column(self):
        columns = self._run(category="funding_agency")
        assert "funding_agency_name" in columns

    def test_funding_subagency_breakdown_adds_column(self):
        columns = self._run(category="funding_subagency")
        assert "funding_sub_agency_name" in columns

    def test_recipient_duns_breakdown_adds_column(self):
        columns = self._run(category="recipient_duns")
        assert "recipient_duns" in columns

    def test_federal_account_breakdown_adds_column(self):
        columns = self._run(category="federal_account")
        assert "federal_accounts_funding_this_award" in columns

    def test_unverified_category_adds_no_column(self):
        """district and defc have no live-verified download column (see
        _DOWNLOAD_COLUMN_BY_CATEGORY's own docstring) - must not guess one."""
        columns = self._run(category="district")
        assert "district" not in columns
        columns = self._run(category="defc")
        assert "disaster_emergency_fund_codes" not in columns

    def test_naics_filter_and_naics_category_together_dont_duplicate_column(self):
        columns = self._run(naics_code="518210", category="naics")
        assert columns.count("naics_code") == 1

    def test_awarding_agency_and_recipient_categories_need_no_extra_column(self):
        """awarding_agency_name/recipient_name are already in the base column set for
        every "awards"/"transactions" download - no entry needed in
        _DOWNLOAD_COLUMN_BY_CATEGORY for these two categories."""
        columns = self._run(category="awarding_agency")
        assert columns.count("awarding_agency_name") == 1
        columns = self._run(category="recipient")
        assert columns.count("recipient_name") == 1


class TestAwardIdEndpoint:
    def test_contract_prefix(self):
        assert _award_id_endpoint("CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-") == "contract"

    def test_idv_prefix(self):
        assert _award_id_endpoint("CONT_IDV_BBGBPA08452513_9568") == "idv"

    def test_assistance_prefix(self):
        assert _award_id_endpoint("ASST_NON_H79TI081692_7522") == "assistance"

    def test_unrecognized_prefix_returns_none(self):
        assert _award_id_endpoint("N0002404C2105") is None




def test_multi_year_intent_scopes_filter_to_the_whole_range_not_just_the_start_year():
    intent = DownloadIntent(agency_raw="NSF", start_year=2022, end_year=2024)
    job = DownloadJobResponse(
        status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
        file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
    )
    finished = DownloadStatusResponse(
        status="finished", file_name="x.zip",
        file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
    )
    client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
    with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
        handle_download_followup(intent.model_dump(), "conv-1")
    assert client.last_filters.time_period[0].start_date == "2021-10-01"
    assert client.last_filters.time_period[0].end_date == "2024-09-30"

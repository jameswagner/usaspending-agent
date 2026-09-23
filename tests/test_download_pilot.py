from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.download_pilot import (
    DownloadIntent,
    _extract_download_intent,
    _is_download_followup,
    _looks_like_download_request,
    _unsupported_download_label,
    handle_download_request,
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
    """Stands in for USASpendingClient - only what handle_download_request calls is faked."""

    def __init__(self, agency=None, job=None, statuses=None, raises=None):
        self._agency = agency
        self._job = job
        self._statuses = list(statuses or [])
        self._raises = raises
        self.last_filters = None

    def find_agency_by_name(self, name):
        return self._agency

    def download_awards(self, filters, columns, file_format="csv"):
        self.last_filters = filters
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


class TestLooksLikeDownloadRequest:
    def test_matches_download_keyword(self):
        assert _looks_like_download_request("Can I download NSF's FY2024 awards?")

    def test_matches_csv_keyword(self):
        assert _looks_like_download_request("Give me a CSV of HHS grants")

    def test_ordinary_question_does_not_match(self):
        assert not _looks_like_download_request("How much did NSF spend in FY2024?")


class TestUnsupportedDownloadLabel:
    def test_transaction_level_is_unsupported(self):
        assert _unsupported_download_label("download transaction-level data for NSF") is not None

    def test_awards_download_is_supported(self):
        assert _unsupported_download_label("download NSF's awards as a CSV") is None


class TestIsDownloadFollowup:
    def test_no_history_is_not_a_followup(self):
        assert not _is_download_followup(None)
        assert not _is_download_followup([])

    def test_prior_download_answer_is_a_followup(self):
        messages = [
            HumanMessage(content="download NSF's January 2024 awards as a CSV"),
            AIMessage(content="Your download is ready: 100 rows in x.zip.\nhttps://..."),
        ]
        assert _is_download_followup(messages)

    def test_ordinary_prior_answer_is_not_a_followup(self):
        messages = [HumanMessage(content="how much did NSF spend in FY2024?"), AIMessage(content="$8.86 billion.")]
        assert not _is_download_followup(messages)


def make_tool_use_response(input_dict):
    block = MagicMock(type="tool_use", input=input_dict)
    return MagicMock(content=[block])


class TestExtractDownloadIntent:
    def test_wants_download_false_returns_none(self):
        # Regression (live-reported): an unrelated question after a download turn served a stale cached file.
        response = make_tool_use_response(
            {"wants_download": False, "agency_raw": "NSF", "start_year": 2024, "end_year": 2024}
        )
        with patch("backend.app.agent.download_pilot._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            result = _extract_download_intent("how is NSF spending broken down by NAICS code in 2024")
        assert result is None

    def test_wants_download_true_returns_intent(self):
        response = make_tool_use_response(
            {"wants_download": True, "agency_raw": "NSF", "start_year": 2024, "end_year": 2024}
        )
        with patch("backend.app.agent.download_pilot._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            result = _extract_download_intent("download NSF's FY2024 awards as a CSV")
        assert result is not None
        assert result.agency_raw == "NSF"


class TestHandleDownloadRequest:
    def test_unsupported_endpoint_returns_fixed_message_without_calling_client(self):
        with patch("backend.app.agent.download_pilot._get_usaspending_client") as get_client:
            result = handle_download_request("download transaction-level data for NSF", "conv-1")
        get_client.assert_not_called()
        assert result is not None
        assert "isn't supported yet" in result.answer_text
        assert result.downloads == []

    def test_ambiguous_intent_falls_through_to_tool_loop(self):
        with patch(
            "backend.app.agent.download_pilot._extract_download_intent", return_value=None
        ):
            result = handle_download_request("download the NSF data", "conv-1")
        assert result is None

    def test_recent_messages_are_forwarded_to_intent_extraction_as_history(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        captured = {}

        def fake_extract(question, history_block=""):
            captured["history_block"] = history_block
            return intent

        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        recent_messages = [
            HumanMessage(content="download NSF's January 2024 awards as a CSV"),
            AIMessage(content="Your download is ready: 1 row in x.zip.\nhttps://..."),
        ]
        with patch("backend.app.agent.download_pilot._extract_download_intent", side_effect=fake_extract), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            handle_download_request("how about February 2024", "conv-1", recent_messages)
        assert "January 2024" in captured["history_block"]

    def test_unresolvable_agency_falls_through_to_tool_loop(self):
        intent = DownloadIntent(agency_raw="Not A Real Agency", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=None)
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            result = handle_download_request("download Not A Real Agency's awards", "conv-1")
        assert result is None

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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
        assert result is not None
        assert len(result.downloads) == 1
        assert result.downloads[0].status == "finished"
        assert result.downloads[0].url == finished.file_url
        assert result.downloads[0].total_rows == 456
        assert len(result.tool_citations) == 1
        assert result.tool_citations[0].tool_name == "download_awards"
        assert result.tool_citations[0].parameters["agency_name"] == "National Science Foundation"

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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            handle_download_request("download NSF's January 2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=january), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client_jan):
            handle_download_request("download NSF's January 2024 awards as a CSV", "conv-1")

        february = DownloadIntent(agency_raw="NSF", start_date="2024-02-01", end_date="2024-02-29")
        client_feb = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=february), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client_feb):
            handle_download_request("download NSF's February 2024 awards as a CSV", "conv-1")

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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_pilot.time.sleep"):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_pilot.time.sleep"):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_pilot.time.sleep"):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
        assert result is not None
        assert "failed to generate" in result.answer_text
        assert "internal error" in result.answer_text
        assert result.downloads[0].status == "failed"

    def test_api_error_returns_error_message(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=make_agency(), raises=USASpendingAPIError("500: boom"))
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
        assert result is not None
        assert "This download failed" in result.answer_text
        assert result.downloads == []

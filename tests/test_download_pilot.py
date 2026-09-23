from unittest.mock import patch

from backend.app.agent.download_pilot import (
    DownloadIntent,
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
    """Stands in for USASpendingClient - only the methods
    handle_download_request actually calls are faked."""

    def __init__(self, agency=None, job=None, statuses=None, raises=None):
        self._agency = agency
        self._job = job
        self._statuses = list(statuses or [])
        self._raises = raises

    def find_agency_by_name(self, name):
        return self._agency

    def download_awards(self, filters, columns, file_format="csv"):
        if self._raises:
            raise self._raises
        return self._job

    def get_download_status(self, file_name):
        if self._raises:
            raise self._raises
        return self._statuses.pop(0)


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
        assert result.downloads[0].url is None

    def test_ready_status_keeps_polling_instead_of_stopping_immediately(self):
        # Regression: a freshly queued job answers "ready" (not "running")
        # for the first several seconds, confirmed live - a poll loop that
        # only recognized "running" as in-progress stopped after one check.
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
        assert result.downloads[0].url is None

    def test_api_error_returns_error_message(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=make_agency(), raises=USASpendingAPIError("500: boom"))
        with patch("backend.app.agent.download_pilot._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_pilot._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
        assert result is not None
        assert "This download failed" in result.answer_text
        assert result.downloads == []

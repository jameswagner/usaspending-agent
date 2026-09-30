from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from backend.app.agent.download_handler import (
    DownloadIntent,
    _download_intent_context,
    _extract_download_intent,
    _extract_prior_tool_context,
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
        self.last_columns = None
        self.last_spending_level = None

    def find_agency_by_name(self, name):
        return self._agency

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


class TestLooksLikeDownloadRequest:
    def test_matches_download_keyword(self):
        assert _looks_like_download_request("Can I download NSF's FY2024 awards?")

    def test_matches_csv_keyword(self):
        assert _looks_like_download_request("Give me a CSV of HHS grants")

    def test_ordinary_question_does_not_match(self):
        assert not _looks_like_download_request("How much did NSF spend in FY2024?")


class TestUnsupportedDownloadLabel:
    def test_account_level_is_unsupported(self):
        assert _unsupported_download_label("download federal account-level data for NSF") is not None

    def test_disaster_data_is_unsupported(self):
        assert _unsupported_download_label("download disaster relief spending data") is not None

    def test_awards_download_is_supported(self):
        assert _unsupported_download_label("download NSF's awards as a CSV") is None

    def test_transaction_level_is_now_supported(self):
        # Regression: transactions used to be a fixed "not supported" label - spending_level covers it now.
        assert _unsupported_download_label("download transaction-level data for NSF") is None

    def test_subaward_download_is_now_supported(self):
        assert _unsupported_download_label("download sub-award data for NSF") is None


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


class TestExtractPriorToolContext:
    """Regression coverage: the download follow-up extractor used to see only lossy
    Q/A prose, which left room to invent fields (e.g. award_type) nothing ever stated. This
    reads the real structured tool-call/stashed-intent history instead."""

    def test_no_messages_returns_none(self):
        assert _extract_prior_tool_context(None) is None
        assert _extract_prior_tool_context([]) is None

    def test_plain_text_only_history_returns_none(self):
        messages = [HumanMessage(content="how much did NSF spend in FY2024?"), AIMessage(content="$8.86 billion.")]
        assert _extract_prior_tool_context(messages) is None

    def test_stashed_download_intent_is_recovered_verbatim(self):
        stashed = {"agency_raw": "National Science Foundation", "spending_level": "transactions", "start_year": 2023, "end_year": 2023}
        messages = [
            HumanMessage(content="download NSF's transactions for FY2023"),
            AIMessage(content="Your download is ready: 30790 rows...", additional_kwargs={"download_intent": stashed}),
        ]
        assert _extract_prior_tool_context(messages) == stashed

    def test_spending_tool_call_args_are_mapped_to_download_fields(self):
        messages = [
            HumanMessage(content="top NAICS codes for NSF in FY2025"),
            AIMessage(content="", tool_calls=[{
                "name": "get_spending_by_category",
                "args": {
                    "category": "naics", "agency_name": "National Science Foundation",
                    "start_year": 2025, "end_year": 2025, "limit": 10, "award_type": "contracts",
                },
                "id": "call_1",
            }]),
            AIMessage(content="Here's the breakdown..."),
        ]
        context = _extract_prior_tool_context(messages)
        assert context == {
            "agency_raw": "National Science Foundation", "start_year": 2025, "end_year": 2025,
            "award_type": "contracts",
        }
        # Display-only args must never leak in as if they were scope filters.
        assert "limit" not in context
        assert "category" not in context

    def test_tool_call_with_no_mappable_fields_returns_none(self):
        messages = [AIMessage(content="", tool_calls=[{"name": "search_guide", "args": {"query": "what is a sub-award?"}, "id": "call_1"}])]
        assert _extract_prior_tool_context(messages) is None

    def test_most_recent_ai_message_wins_over_an_older_one(self):
        messages = [
            AIMessage(content="", tool_calls=[{
                "name": "get_spending_by_category",
                "args": {"agency_name": "Department of Energy", "start_year": 2022, "end_year": 2022},
                "id": "call_1",
            }]),
            AIMessage(content="Your download is ready: ...", additional_kwargs={
                "download_intent": {"agency_raw": "National Science Foundation", "spending_level": "awards"}
            }),
        ]
        context = _extract_prior_tool_context(messages)
        assert context == {"agency_raw": "National Science Foundation", "spending_level": "awards"}


class TestDownloadIntentContext:
    def test_year_range_intent(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024, spending_level="transactions")
        context = _download_intent_context(intent, "National Science Foundation")
        assert context == {
            "spending_level": "transactions", "time_period_type": "fiscal",
            "agency_raw": "National Science Foundation", "start_year": 2024, "end_year": 2024,
        }

    def test_date_range_intent_omits_start_end_year(self):
        intent = DownloadIntent(agency_raw="NSF", start_date="2024-01-01", end_date="2024-01-31")
        context = _download_intent_context(intent, "National Science Foundation")
        assert context["start_date"] == "2024-01-01"
        assert context["end_date"] == "2024-01-31"
        assert "start_year" not in context

    def test_award_type_only_included_when_set(self):
        intent = DownloadIntent(start_year=2024, end_year=2024)
        assert "award_type" not in _download_intent_context(intent, None)


def make_tool_use_response(input_dict):
    block = MagicMock(type="tool_use", input=input_dict)
    return MagicMock(content=[block])


class TestExtractDownloadIntent:
    def test_wants_download_false_returns_none(self):
        # Regression (live-reported): an unrelated question after a download turn served a stale cached file.
        response = make_tool_use_response(
            {"wants_download": False, "agency_raw": "NSF", "start_year": 2024, "end_year": 2024}
        )
        with patch("backend.app.agent.download_handler._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            result = _extract_download_intent("how is NSF spending broken down by NAICS code in 2024")
        assert result is None

    def test_wants_download_true_returns_intent(self):
        response = make_tool_use_response(
            {"wants_download": True, "agency_raw": "NSF", "start_year": 2024, "end_year": 2024}
        )
        with patch("backend.app.agent.download_handler._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            result = _extract_download_intent("download NSF's FY2024 awards as a CSV")
        assert result is not None
        assert result.agency_raw == "NSF"

    def test_prior_context_is_passed_to_the_model_as_ground_truth(self):
        # Regression: a vague "carry over from history" instruction let the model invent
        # an award_type nothing ever stated. With prior_context, the system prompt must both
        # supply the exact prior values and explicitly forbid guessing anything beyond them.
        response = make_tool_use_response(
            {"wants_download": True, "agency_raw": "NSF", "start_year": 2024, "end_year": 2024}
        )
        prior_context = {"agency_raw": "National Science Foundation", "spending_level": "transactions", "start_year": 2023, "end_year": 2023}
        with patch("backend.app.agent.download_handler._get_client") as get_client:
            get_client.return_value.messages.create.return_value = response
            _extract_download_intent("how about FY2024", prior_context=prior_context)
        call_kwargs = get_client.return_value.messages.create.call_args.kwargs
        assert "National Science Foundation" in call_kwargs["system"]
        assert "do not guess or default" in call_kwargs["system"].lower()


class TestHandleDownloadRequest:
    def test_unsupported_endpoint_returns_fixed_message_without_calling_client(self):
        with patch("backend.app.agent.download_handler._get_usaspending_client") as get_client:
            result = handle_download_request("download disaster relief spending data", "conv-1")
        get_client.assert_not_called()
        assert result is not None
        assert "isn't supported yet" in result.answer_text
        assert result.downloads == []

    def test_ambiguous_intent_falls_through_to_tool_loop(self):
        with patch(
            "backend.app.agent.download_handler._extract_download_intent", return_value=None
        ):
            result = handle_download_request("download the NSF data", "conv-1")
        assert result is None

    def test_recent_messages_are_forwarded_to_intent_extraction_as_history(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        captured = {}

        def fake_extract(question, history_block="", prior_context=None):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", side_effect=fake_extract), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_request("how about February 2024", "conv-1", recent_messages)
        assert "January 2024" in captured["history_block"]

    def test_prior_tool_context_is_forwarded_to_intent_extraction(self):
        # Regression: the resolved structured context from a prior turn (not just prose)
        # must reach the extractor, so it has real values instead of having to guess.
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024, spending_level="transactions")
        captured = {}

        def fake_extract(question, history_block="", prior_context=None):
            captured["prior_context"] = prior_context
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
            HumanMessage(content="download NSF's transactions for FY2023"),
            AIMessage(content="Your download is ready: 30790 rows...", additional_kwargs={
                "download_intent": {"agency_raw": "National Science Foundation", "spending_level": "transactions", "start_year": 2023, "end_year": 2023}
            }),
        ]
        with patch("backend.app.agent.download_handler._extract_download_intent", side_effect=fake_extract), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_request("how about FY2024", "conv-1", recent_messages)
        assert captured["prior_context"] == {
            "agency_raw": "National Science Foundation", "spending_level": "transactions",
            "start_year": 2023, "end_year": 2023,
        }

    def test_result_carries_resolved_intent_context_for_the_next_followup(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024, spending_level="transactions")
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=1,
        )
        client = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's transactions for FY2024", "conv-1")
        assert result.download_intent_context == {
            "spending_level": "transactions", "time_period_type": "fiscal",
            "agency_raw": "National Science Foundation", "start_year": 2024, "end_year": 2024,
        }
        # No award_type was ever set - must not appear, so a later follow-up has nothing to
        # mistakenly "carry over".
        assert "award_type" not in result.download_intent_context

    def test_unresolvable_agency_falls_through_to_tool_loop(self):
        intent = DownloadIntent(agency_raw="Not A Real Agency", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=None)
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=january), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client_jan):
            handle_download_request("download NSF's January 2024 awards as a CSV", "conv-1")

        february = DownloadIntent(agency_raw="NSF", start_date="2024-02-01", end_date="2024-02-29")
        client_feb = FakeDownloadClient(agency=make_agency(), job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=february), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client_feb):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client), \
             patch("backend.app.agent.download_handler.time.sleep"):
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
        assert result is not None
        assert "failed to generate" in result.answer_text
        assert "internal error" in result.answer_text
        assert result.downloads[0].status == "failed"

    def test_api_error_returns_error_message(self):
        intent = DownloadIntent(agency_raw="NSF", start_year=2024, end_year=2024)
        client = FakeDownloadClient(agency=make_agency(), raises=USASpendingAPIError("500: boom"))
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1")
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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request(f"download NSF's FY2024 {spending_level}", "conv-1")
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

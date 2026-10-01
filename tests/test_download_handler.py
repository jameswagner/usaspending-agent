from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.app.agent.download_handler import (
    DownloadIntent,
    _award_id_endpoint,
    _download_intent_context,
    _extract_award_id_from_history,
    _extract_download_intent,
    _extract_prior_tool_context,
    _is_download_followup,
    _looks_like_download_request,
    _resolve_single_award_id,
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
            AIMessage(
                content="Your download is ready: 100 rows in x.zip.\nhttps://...",
                additional_kwargs={"download_intent": {"agency_raw": "National Science Foundation"}},
            ),
        ]
        assert _is_download_followup(messages)

    def test_ordinary_prior_answer_is_not_a_followup(self):
        messages = [HumanMessage(content="how much did NSF spend in FY2024?"), AIMessage(content="$8.86 billion.")]
        assert not _is_download_followup(messages)

    def test_download_shaped_answer_without_a_stashed_intent_is_not_a_followup(self):
        # An unsupported-endpoint refusal reads like a download answer but resolved no intent,
        # so a follow-up has nothing to carry over from it.
        messages = [
            HumanMessage(content="can I get a CSV of NSF's account-level data?"),
            AIMessage(content="Downloading account-level data isn't supported yet — only award-level CSV exports are, for now."),
        ]
        assert not _is_download_followup(messages)

    def test_detection_survives_rewording_the_download_answer(self):
        messages = [
            HumanMessage(content="download NSF's January 2024 awards as a CSV"),
            AIMessage(
                content="Here you go, 100 rows: https://...",
                additional_kwargs={"download_intent": {"agency_raw": "National Science Foundation"}},
            ),
        ]
        assert _is_download_followup(messages)


class TestExtractPriorToolContext:
    """Regression coverage: the download follow-up extractor used to see only lossy
    Q/A prose, which left room to invent fields (e.g. award_type) nothing ever stated. This
    reads the real structured tool-call/stashed-intent history instead."""

    def test_no_messages_returns_none(self):
        assert _extract_prior_tool_context(None) == (None, [])
        assert _extract_prior_tool_context([]) == (None, [])

    def test_plain_text_only_history_returns_none(self):
        messages = [HumanMessage(content="how much did NSF spend in FY2024?"), AIMessage(content="$8.86 billion.")]
        assert _extract_prior_tool_context(messages) == (None, [])

    def test_stashed_download_intent_is_recovered_verbatim(self):
        stashed = {"agency_raw": "National Science Foundation", "spending_level": "transactions", "start_year": 2023, "end_year": 2023}
        messages = [
            HumanMessage(content="download NSF's transactions for FY2023"),
            AIMessage(content="Your download is ready: 30790 rows...", additional_kwargs={"download_intent": stashed}),
        ]
        assert _extract_prior_tool_context(messages) == (stashed, [])

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
        context, dropped = _extract_prior_tool_context(messages)
        assert context == {
            "agency_raw": "National Science Foundation", "start_year": 2025, "end_year": 2025,
            "award_type": "contracts",
        }
        assert dropped == []
        # Display-only args must never leak in as if they were scope filters.
        assert "limit" not in context
        assert "category" not in context

    def test_tool_call_with_no_mappable_fields_returns_none(self):
        messages = [AIMessage(content="", tool_calls=[{"name": "search_guide", "args": {"query": "what is a sub-award?"}, "id": "call_1"}])]
        assert _extract_prior_tool_context(messages) == (None, [])

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
        context, dropped = _extract_prior_tool_context(messages)
        assert context == {"agency_raw": "National Science Foundation", "spending_level": "awards"}
        assert dropped == []

    def test_naics_scoped_category_call_is_carried_into_download_fields(self):
        """A NAICS/PSC-scoped answer's scope must survive into the
        download follow-up, not just agency/time_period/award_type."""
        messages = [
            HumanMessage(content="how much did NSF spend on cloud computing in FY2024?"),
            AIMessage(content="", tool_calls=[{
                "name": "get_spending_over_time",
                "args": {
                    "agency_name": "National Science Foundation", "start_year": 2024, "end_year": 2024,
                    "naics_code": "518210", "group": "fiscal_year",
                },
                "id": "call_1",
            }]),
            AIMessage(content="NSF spent $12.3 million on cloud computing infrastructure in FY2024."),
        ]
        context, dropped = _extract_prior_tool_context(messages)
        assert context == {
            "agency_raw": "National Science Foundation", "start_year": 2024, "end_year": 2024,
            "naics_code": "518210",
        }
        assert dropped == []
        assert "group" not in context

    def test_recipient_id_scope_is_reported_as_dropped_not_silently_carried(self):
        """recipient_id has no equivalent in /api/v2/download/search/'s own Filters object -
        live-verified 2026-09-30 (a bogus recipient_id alongside a real scope produced the
        exact same job as omitting it entirely). Carrying it forward would silently widen
        the download past the answer it continues, so it must come back as a caveat, not
        a mapped field."""
        messages = [
            HumanMessage(content="how much has this recipient received from HHS in FY2024?"),
            AIMessage(content="", tool_calls=[{
                "name": "get_spending_over_time",
                "args": {
                    "agency_name": "Department of Health and Human Services",
                    "recipient_id": "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P",
                    "start_year": 2024, "end_year": 2024,
                },
                "id": "call_1",
            }]),
            AIMessage(content="This recipient received $4.2 million from HHS in FY2024."),
        ]
        context, dropped = _extract_prior_tool_context(messages)
        assert "recipient_id" not in context
        assert dropped == ["the recipient ID filter"]


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
            AIMessage(
                content="Your download is ready: 1 row in x.zip.\nhttps://...",
                additional_kwargs={"download_intent": {"agency_raw": "National Science Foundation"}},
            ),
        ]
        with patch("backend.app.agent.download_handler._extract_download_intent", side_effect=fake_extract), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_request("how about February 2024", "conv-1", recent_messages)
        assert "January 2024" in captured["history_block"]

    def test_history_is_withheld_when_the_prior_turn_was_not_a_download(self):
        # The history block's prompt asserts the user is continuing a prior download, so an
        # ordinary preceding answer must not get it - it would invite carrying over that turn's scope.
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
            HumanMessage(content="how much did NSF spend on cloud computing in FY2024?"),
            AIMessage(content="$1.2 billion."),
        ]
        with patch("backend.app.agent.download_handler._extract_download_intent", side_effect=fake_extract), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_request("download NSF's FY2024 awards as a CSV", "conv-1", recent_messages)
        assert captured["history_block"] == ""

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
        with patch("backend.app.agent.download_handler._extract_download_intent", return_value=intent), \
             patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            handle_download_request("download NSF's FY2024 awards", "conv-1")
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

    def _run(self, naics_code=None, psc_code=None, spending_level="awards"):
        intent = DownloadIntent(
            agency_raw="NSF", start_year=2024, end_year=2024,
            naics_code=naics_code, psc_code=psc_code, spending_level=spending_level,
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
            handle_download_request("download NSF's FY2024 awards", "conv-1")
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


class TestAwardIdEndpoint:
    def test_contract_prefix(self):
        assert _award_id_endpoint("CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-") == "contract"

    def test_idv_prefix(self):
        assert _award_id_endpoint("CONT_IDV_BBGBPA08452513_9568") == "idv"

    def test_assistance_prefix(self):
        assert _award_id_endpoint("ASST_NON_H79TI081692_7522") == "assistance"

    def test_unrecognized_prefix_returns_none(self):
        assert _award_id_endpoint("N0002404C2105") is None


class TestExtractAwardIdFromHistory:
    def test_no_history_returns_none(self):
        assert _extract_award_id_from_history(None) is None
        assert _extract_award_id_from_history([]) is None

    def test_finds_internal_id_in_tool_message(self):
        messages = [
            HumanMessage(content="find NASA's biggest contract in 2024"),
            ToolMessage(
                content="CONT_AWD_X — Boeing: $1.2M [internal_id: CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-]",
                tool_call_id="1",
            ),
            AIMessage(content="NASA's biggest contract in 2024 was with Boeing for $1.2M."),
        ]
        assert (
            _extract_award_id_from_history(messages)
            == "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-"
        )

    def test_ignores_ai_message_text(self):
        # Only ToolMessage content is searched - an AI message shouldn't match.
        messages = [AIMessage(content="internal_id: CONT_AWD_looks_like_one_but_isnt_tagged")]
        assert _extract_award_id_from_history(messages) is None

    def test_most_recent_tool_message_wins(self):
        messages = [
            ToolMessage(content="[internal_id: CONT_AWD_OLDER]", tool_call_id="1"),
            ToolMessage(content="[internal_id: CONT_AWD_NEWER]", tool_call_id="2"),
        ]
        assert _extract_award_id_from_history(messages) == "CONT_AWD_NEWER"


class TestResolveSingleAwardId:
    def test_award_id_in_question_is_used_directly(self):
        assert (
            _resolve_single_award_id("download CONT_IDV_BBGBPA08452513_9568", None)
            == "CONT_IDV_BBGBPA08452513_9568"
        )

    def test_followup_phrase_without_history_returns_none(self):
        assert _resolve_single_award_id("download this award", None) is None

    def test_followup_phrase_resolves_from_history(self):
        messages = [ToolMessage(content="[internal_id: ASST_NON_H79TI081692_7522]", tool_call_id="1")]
        assert _resolve_single_award_id("download this grant", messages) == "ASST_NON_H79TI081692_7522"

    def test_ordinary_multi_award_question_returns_none(self):
        assert _resolve_single_award_id("download NSF's FY2024 awards as a CSV", None) is None


class TestHandleDownloadRequestSingleAward:
    def test_direct_award_id_routes_to_contract_endpoint(self):
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip", total_rows=None,
        )
        client = FakeDownloadClient(job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request(
                "download CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-", "conv-1"
            )
        assert result is not None
        assert client.last_award_id == "CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-"
        assert result.downloads[0].status == "finished"
        assert result.tool_citations[0].tool_name == "download_contract"

    def test_direct_idv_award_id_routes_to_idv_endpoint(self):
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        client = FakeDownloadClient(job=job, statuses=[finished])
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download CONT_IDV_BBGBPA08452513_9568", "conv-1")
        assert result.tool_citations[0].tool_name == "download_idv"

    def test_followup_phrase_uses_history_resolved_award_id(self):
        job = DownloadJobResponse(
            status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
            file_name="x.zip", file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        finished = DownloadStatusResponse(
            status="finished", file_name="x.zip",
            file_url="https://files.usaspending.gov/generated_downloads/x.zip",
        )
        client = FakeDownloadClient(job=job, statuses=[finished])
        recent_messages = [
            HumanMessage(content="get me the details on that contract"),
            ToolMessage(content="[internal_id: ASST_NON_H79TI081692_7522]", tool_call_id="1"),
            AIMessage(content="Here are the details."),
        ]
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request("download this grant", "conv-1", recent_messages)
        assert client.last_award_id == "ASST_NON_H79TI081692_7522"
        assert result.tool_citations[0].tool_name == "download_assistance"

    def test_followup_phrase_with_no_resolvable_id_returns_helpful_message_not_none(self):
        result = handle_download_request("download this award", "conv-1", None)
        assert result is not None
        assert "conv-1" == result.conversation_id
        assert result.downloads == []

    def test_api_error_on_single_award_download_returns_error_message(self):
        client = FakeDownloadClient(raises=USASpendingAPIError("500: boom"))
        with patch("backend.app.agent.download_handler._get_usaspending_client", return_value=client):
            result = handle_download_request(
                "download CONT_AWD_N0002404C2105_9700_-NONE-_-NONE-", "conv-1"
            )
        assert "This download failed" in result.answer_text

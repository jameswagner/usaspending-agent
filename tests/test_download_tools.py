import asyncio
import inspect
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from backend.app.agent.langgraph_tools import LANGGRAPH_TOOLS
from backend.app.agent.orchestrator import _build_result
from backend.app.agent.response_shaping import DownloadSpec
from backend.app.agent.streaming import _build_done_payload, sse_event_generator
from backend.app.agent.tool_filters import SpendingFilterParams
from backend.app.agent.tools import (
    _tool_call_log,
    download_records,
    download_single_award,
)
from backend.app.usaspending import DownloadJobResponse, DownloadStatusResponse

JOB = DownloadJobResponse(
    status_url="https://api.usaspending.gov/api/v2/download/status?file_name=x.zip",
    file_name="x.zip",
    file_url="https://files.usaspending.gov/generated_downloads/x.zip",
)
STATUS = DownloadStatusResponse(status="finished", file_name="x.zip", file_url=JOB.file_url, total_rows=12)


def test_records_project_to_result_and_sse_with_scope_citation():
    client = MagicMock()
    client.find_agency_by_name.return_value = SimpleNamespace(agency_name="National Science Foundation")
    client.download_search.return_value = JOB
    filters = MagicMock()
    filters.model_dump.return_value = {"keywords": ["information technology"]}
    token = _tool_call_log.set([])
    try:
        with patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client), patch(
            "backend.app.agent.tools.download._build_filters", return_value=filters
        ) as build_filters, patch(
            "backend.app.agent.tools.download._poll_until_finished", return_value=STATUS
        ), patch("backend.app.agent.tools.download._pop_naics_disclosure", return_value=None):
            text = download_records.func(
                agency_name="NSF", start_fiscal_year=2024, end_fiscal_year=2024,
                award_type="contracts", keywords="information technology",
            )
        assert "12 rows" in text
        assert build_filters.call_args.kwargs["keywords"] == "information technology"
        assert build_filters.call_args.kwargs["award_type"] == "contracts"
        assert client.download_search.call_count == 1
        result = _build_result("The file is ready.", "conversation")
    finally:
        _tool_call_log.reset(token)
    assert len(result.downloads) == 1
    assert result.downloads[0].file_name == "x.zip"
    assert result.charts == []
    assert len(result.tool_citations) == 1
    assert result.tool_citations[0].parameters["keywords"] == "information technology"
    assert '"keywords": ["information technology"]' in result.tool_citations[0].curl
    payload = _build_done_payload(result)
    assert payload["downloads"] == [result.downloads[0].model_dump()]
    assert payload["tool_citations"] == [result.tool_citations[0].model_dump()]


def test_single_award_uses_prefix_endpoint_and_projects_download():
    client = MagicMock()
    client.download_assistance.return_value = JOB
    token = _tool_call_log.set([])
    try:
        with patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client), patch(
            "backend.app.agent.tools.download._poll_until_finished", return_value=STATUS
        ):
            download_single_award.func("ASST_VALID_ID")
        result = _build_result("ready", "conversation")
    finally:
        _tool_call_log.reset(token)
    client.download_assistance.assert_called_once_with("ASST_VALID_ID")
    assert len(result.downloads) == 1
    assert "ASST_VALID_ID" in result.tool_citations[0].curl


def test_unsupported_recipient_id_cannot_queue_a_wider_file():
    client = MagicMock()
    with patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client):
        text = download_records.func(
            start_fiscal_year=2024, end_fiscal_year=2024, recipient_id="recipient-1",
        )
    assert "cannot filter by recipient ID" in text
    client.download_search.assert_not_called()


def test_two_psc_codes_share_one_awards_only_download_job():
    client = MagicMock()
    client.download_search.return_value = JOB
    filters = MagicMock()
    with patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client), patch(
        "backend.app.agent.tools.download._build_filters", return_value=filters
    ), patch("backend.app.agent.tools.download._poll_until_finished", return_value=STATUS), patch(
        "backend.app.agent.tools.download._pop_naics_disclosure", return_value=None
    ):
        download_records.func(
            start_fiscal_year=2024, end_fiscal_year=2024, award_type="contracts",
            psc_codes=["da01", "D302", "DA01"], include_subawards=False,
        )
    assert filters.psc_codes == ["DA01", "D302"]
    assert client.download_search.call_count == 1
    assert client.download_search.call_args.args[2] == ["awards"]
    assert client.download_search.call_args.args[2] == ["awards"]


def test_psc_code_and_codes_cannot_be_combined():
    client = MagicMock()
    with patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client):
        text = download_records.func(
            start_fiscal_year=2024, end_fiscal_year=2024, psc_code="DA01", psc_codes=["D302"],
        )
    assert "either psc_code" in text
    client.download_search.assert_not_called()


def test_unrelated_download_spec_is_not_projected():
    token = _tool_call_log.set([
        ("other_tool", DownloadSpec(file_name="x.zip", url=JOB.file_url, status_url=JOB.status_url, status="finished"), {})
    ])
    try:
        result = _build_result("ok", "conversation")
    finally:
        _tool_call_log.reset(token)
    assert result.downloads == []


def test_download_schema_covers_shared_spending_filters():
    actual = set(inspect.signature(download_records.func).parameters)
    assert set(SpendingFilterParams.__annotations__) <= actual


def test_real_graph_stream_done_contains_download_and_citation():
    class ScriptedModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    model = ScriptedModel(responses=[
        AIMessage(content="", tool_calls=[{
            "name": "download_records",
            "args": {"agency_name": "NSF", "start_fiscal_year": 2024, "end_fiscal_year": 2024},
            "id": "download-1",
        }]),
        AIMessage(content="The CSV is ready."),
    ])
    graph = create_react_agent(
        model=model,
        tools=[tool for tool in LANGGRAPH_TOOLS if tool.name == "download_records"],
        checkpointer=InMemorySaver(),
    )
    client = MagicMock()
    client.find_agency_by_name.return_value = SimpleNamespace(agency_name="National Science Foundation")
    client.download_search.return_value = JOB
    filters = MagicMock()
    filters.model_dump.return_value = {"time_period": [{"start_date": "2023-10-01", "end_date": "2024-09-30"}]}

    async def collect():
        return [frame async for frame in sse_event_generator("probe", "probe")]

    with patch("backend.app.agent.streaming._get_conversation_graph", return_value=graph), patch(
        "backend.app.agent.streaming._is_in_scope", return_value=True
    ), patch("backend.app.agent.tools.download._get_usaspending_client", return_value=client), patch(
        "backend.app.agent.tools.download._build_filters", return_value=filters
    ), patch("backend.app.agent.tools.download._poll_until_finished", return_value=STATUS), patch(
        "backend.app.agent.tools.download._pop_naics_disclosure", return_value=None
    ):
        frames = asyncio.run(collect())

    done = next(json.loads(frame.decode().split("data: ", 1)[1]) for frame in frames if frame.startswith(b"event: done"))
    assert done["downloads"][0]["file_name"] == "x.zip"
    assert done["tool_citations"][0]["tool_name"] == "download_records"

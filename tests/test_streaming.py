import json

from backend.app.agent.orchestrator import NOT_FOUND_MESSAGE, AgentResult
from backend.app.agent.response_shaping import ChartSpec, Citation, ToolCitation
from backend.app.agent.streaming import (
    _build_done_payload,
    _format_sse_frame,
    _summarize_tool_result,
)


class TestFormatSSEFrame:
    def test_basic_shape(self):
        frame = _format_sse_frame("tool_call_start", {"tool_name": "search_awards", "args": {"a": 1}})
        assert frame == b'event: tool_call_start\ndata: {"tool_name": "search_awards", "args": {"a": 1}}\n\n'

    def test_embedded_newline_in_a_string_field_stays_on_one_data_line(self):
        # SSE treats a bare \n as ending the data: field - json.dumps must
        # escape it (\\n) so a multi-line tool summary/error message still
        # round-trips as a single data: line.
        frame = _format_sse_frame("tool_error", {"message": "line one\nline two"})
        text = frame.decode()
        lines = text.split("\n")
        # exactly one data: line, plus the trailing blank lines from \n\n
        data_lines = [line for line in lines if line.startswith("data: ")]
        assert len(data_lines) == 1
        assert json.loads(data_lines[0][len("data: ") :]) == {"message": "line one\nline two"}

    def test_round_trips_through_json(self):
        payload = {"tool_name": "get_spending_by_category", "summary": "NAICS 541511: $90,951,557.61"}
        frame = _format_sse_frame("tool_result", payload)
        _, _, data_part = frame.decode().partition("data: ")
        assert json.loads(data_part.strip()) == payload


class TestSummarizeToolResult:
    def test_plain_text_passed_through(self):
        assert _summarize_tool_result("12 awards found for NSF, FY2023") == "12 awards found for NSF, FY2023"

    def test_strips_untrusted_data_wrapper(self):
        wrapped = "<untrusted_data>\nFY2024: budgetary resources $30,051,004,236.67\n</untrusted_data>"
        assert _summarize_tool_result(wrapped) == "FY2024: budgetary resources $30,051,004,236.67"

    def test_collapses_internal_whitespace_and_newlines(self):
        assert _summarize_tool_result("line one\n  line two\n\nline three") == "line one line two line three"

    def test_truncates_long_content(self):
        long_text = "x" * 300
        result = _summarize_tool_result(long_text)
        assert len(result) <= 201  # _SUMMARY_MAX_CHARS + ellipsis char
        assert result.endswith("…")


class TestBuildDonePayload:
    def test_agent_source_type(self):
        result = AgentResult(answer_text="NSF's budget is $9 billion.", conversation_id="conv-1")
        payload = _build_done_payload(result)
        assert payload["source_type"] == "agent"
        assert payload["answer_text"] == result.answer_text
        assert payload["conversation_id"] == "conv-1"
        assert payload["charts"] == []
        assert payload["citations"] == []
        assert payload["tool_citations"] == []

    def test_not_found_source_type(self):
        result = AgentResult(answer_text=NOT_FOUND_MESSAGE, conversation_id="conv-2")
        payload = _build_done_payload(result)
        assert payload["source_type"] == "not_found"

    def test_charts_citations_tool_citations_are_dumped_to_plain_dicts(self):
        result = AgentResult(
            answer_text="ok",
            conversation_id="conv-3",
            charts=[ChartSpec(chart_type="bar", title="t", labels=["a"], values=[1.0])],
            citations=[Citation(chunk_id="c1", source="Analyst's Guide", page=1)],
            tool_citations=[ToolCitation(tool_name="lookup_agency", parameters={"name": "NSF"}, description="d")],
        )
        payload = _build_done_payload(result)
        assert payload["charts"] == [result.charts[0].model_dump()]
        assert payload["citations"] == [result.citations[0].model_dump()]
        assert payload["tool_citations"] == [result.tool_citations[0].model_dump()]

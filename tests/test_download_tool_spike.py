from types import SimpleNamespace

from backend.app.agent import download_tool_spike as spike


def _job():
    return SimpleNamespace(file_name="job.zip", status_url="https://example.org/status")


def _status():
    return SimpleNamespace(
        file_name="job.zip", file_url="https://example.org/job.zip",
        status="finished", total_rows=7,
    )


def test_shape_a_rejects_mixed_award_and_record_scope(monkeypatch):
    monkeypatch.setattr(spike, "_get_usaspending_client", lambda: None)
    result = spike.download_data.func(award_id="CONT_AWD_X", agency_name="NSF")
    assert "cannot be combined" in result


def test_single_award_dispatches_by_prefix(monkeypatch):
    calls = []
    client = SimpleNamespace(download_contract=lambda award_id: calls.append(award_id) or _job())
    monkeypatch.setattr(spike, "_get_usaspending_client", lambda: client)
    monkeypatch.setattr(spike, "_poll_until_finished", lambda client, name: _status())
    monkeypatch.setattr(spike, "_record_tool_call", lambda name, result, context: calls.append(context))
    assert "CSV download ready" in spike.download_single_award.func("CONT_AWD_X")
    assert calls[0] == "CONT_AWD_X"
    assert calls[1]["endpoint"] == "contract"


def test_records_reuse_filter_and_level_mapping(monkeypatch):
    calls = []
    filters = SimpleNamespace(time_period=None)
    client = SimpleNamespace(
        find_agency_by_name=lambda name: SimpleNamespace(agency_name="National Science Foundation"),
        download_search=lambda filters, columns, levels: calls.append((filters, columns, levels)) or _job(),
    )
    monkeypatch.setattr(spike, "_get_usaspending_client", lambda: client)
    monkeypatch.setattr(spike, "_build_filters", lambda *args, **kwargs: filters)
    monkeypatch.setattr(spike, "_poll_until_finished", lambda client, name: _status())
    monkeypatch.setattr(spike, "_record_tool_call", lambda *args: None)
    result = spike.download_records.func(
        agency_name="NSF", start_fiscal_year=2024, end_fiscal_year=2024,
        spending_level="transactions",
    )
    assert "CSV download ready" in result
    assert calls[0][0] is filters
    assert calls[0][1] == spike._DOWNLOAD_COLUMNS_BY_LEVEL["transactions"]
    assert calls[0][2] == spike._SPENDING_LEVEL_TO_API_ARRAY["transactions"]

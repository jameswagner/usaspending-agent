from __future__ import annotations

from unittest.mock import patch

from backend.app.agent.tools.spending import consolidated
from backend.app.agent.tools.spending.consolidated import query_spending

_BASE = {"start_year": 2024, "end_year": 2024, "agency_name": "National Science Foundation"}


def _call(**kwargs):
    return query_spending.func(**kwargs) if hasattr(query_spending, "func") else query_spending(**kwargs)


def test_aggregate_without_award_type_covers_all_award_types():
    with patch.object(consolidated, "get_spending_over_time", return_value="ok") as agg:
        _call(level="award", group_by="time", **_BASE)
    assert agg.call_args.kwargs["award_type"] is None


def test_aggregate_with_award_type_keeps_it():
    with patch.object(consolidated, "get_spending_over_time", return_value="ok") as agg:
        _call(level="award", group_by="time", award_type="grants", **_BASE)
    assert agg.call_args.kwargs["award_type"] == "grants"


def test_records_without_award_type_are_rejected_not_defaulted():
    with patch.object(consolidated, "search_awards", return_value="ok") as rec:
        result = _call(level="award", **_BASE)
    assert result.startswith("This query failed: award_type is required for record searches")
    rec.assert_not_called()


def test_records_with_award_type_keep_it():
    with patch.object(consolidated, "search_awards", return_value="ok") as rec:
        _call(level="award", award_type="grants", **_BASE)
    assert rec.call_args.kwargs["award_type"] == "grants"

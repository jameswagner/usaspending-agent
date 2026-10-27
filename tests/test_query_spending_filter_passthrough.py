from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.app.agent.tools.spending import category, consolidated
from backend.app.agent.tools.spending.consolidated import query_spending

_BASE = {"start_year": 2025, "end_year": 2025}


def _call(**kwargs):
    return query_spending.func(**kwargs) if hasattr(query_spending, "func") else query_spending(**kwargs)


@pytest.mark.parametrize("group_by, target", [("recipient", "get_spending_by_category"), ("time", "get_spending_over_time")])
def test_breakdowns_keep_account_and_contract_filters(group_by, target):
    with patch.object(consolidated, target, return_value="ok") as agg:
        _call(level="award", group_by=group_by, federal_account="080-0124", set_aside_type=["small_business_set_aside_total"], **_BASE)
    assert agg.call_args.kwargs["federal_account"] == "080-0124"
    assert agg.call_args.kwargs["set_aside_type"] == ["small_business_set_aside_total"]


def test_geography_refuses_filters_it_cannot_apply():
    with patch.object(consolidated, "get_spending_by_geography", return_value="ok") as agg:
        result = _call(level="award", group_by="geography", scope="place_of_performance", geo_layer="state", federal_account="080-0124", **_BASE)
    assert "federal_account" in result and result.startswith("This query failed")
    agg.assert_not_called()


@pytest.mark.parametrize("level, kwargs, named", [
    ("subaward", {"federal_account": "080-0124"}, "federal_account"),
    ("transaction", {"set_aside_type": ["small_business_set_aside_total"]}, "set_aside_type"),
])
def test_records_refuse_filters_their_level_cannot_apply(level, kwargs, named):
    result = _call(level=level, award_type="contracts", **kwargs, **_BASE)
    assert result.startswith("This query failed") and named in result


def test_category_sends_the_federal_account_to_the_api():
    client = MagicMock()
    with patch.object(category, "_get_usaspending_client", return_value=client):
        category.get_spending_by_category_raw("recipient", None, "fiscal", 2025, 2025, federal_account="080-0124")
    filters = client.spending_by_category.call_args.args[1]
    assert [(c.aid, c.main) for c in filters.treasury_account_components] == [("080", "0124")]


def test_records_refuse_recipient_id():
    with patch.object(consolidated, "search_awards", return_value="ok") as rec:
        result = _call(level="award", award_type="contracts", recipient_id="abc-C", **_BASE)
    assert result.startswith("This query failed") and "recipient_id" in result
    rec.assert_not_called()


def test_transactions_refuse_an_outlays_sort_instead_of_sorting_by_amount():
    with patch.object(consolidated, "search_transactions", return_value="ok") as rec:
        result = _call(level="transaction", award_type="contracts", sort_by="outlays", **_BASE)
    assert result.startswith("This query failed") and "outlays" in result
    rec.assert_not_called()


def test_geography_refuses_the_subaward_level():
    with patch.object(consolidated, "get_spending_by_geography", return_value="ok") as agg:
        result = _call(level="subaward", group_by="geography", scope="place_of_performance", geo_layer="state", agency_name="NASA", **_BASE)
    assert result.startswith("This query failed") and "subaward" in result
    agg.assert_not_called()

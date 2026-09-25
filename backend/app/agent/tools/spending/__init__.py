"""The _build_filters-based cluster: get_spending_by_category,
get_spending_over_time, search_awards, search_subawards, search_transactions,
get_spending_by_geography. All six funnel their filter params through the
shared tool_filters._build_filters. Each tool lives in its own module
(category.py, over_time.py, search.py, geography.py); this file re-exports
their public names so existing import sites don't need to change.

query_spending (consolidated.py) is the model-facing tool built on top of
these six - the six themselves stay internal, no longer bound to the agent
directly (see langgraph_tools.py).
"""
from __future__ import annotations

from .category import (
    VALID_CATEGORIES,
    Category,
    _normalize_category,
    get_spending_by_category,
    get_spending_by_category_raw,
)
from .consolidated import Level, SpendingGroupBy, query_spending
from .geography import (
    _format_geography_result,
    get_spending_by_geography,
    get_spending_by_geography_raw,
)
from .over_time import (
    VALID_GROUPS,
    Group,
    _normalize_group,
    get_spending_over_time,
    get_spending_over_time_raw,
)
from .search import (
    search_awards,
    search_awards_raw,
    search_subawards,
    search_subawards_raw,
    search_transactions,
    search_transactions_raw,
)

__all__ = [
    "VALID_CATEGORIES",
    "VALID_GROUPS",
    "Category",
    "Group",
    "Level",
    "SpendingGroupBy",
    "_format_geography_result",
    "_normalize_category",
    "_normalize_group",
    "get_spending_by_category",
    "get_spending_by_category_raw",
    "get_spending_by_geography",
    "get_spending_by_geography_raw",
    "get_spending_over_time",
    "get_spending_over_time_raw",
    "query_spending",
    "search_awards",
    "search_awards_raw",
    "search_subawards",
    "search_subawards_raw",
    "search_transactions",
    "search_transactions_raw",
]

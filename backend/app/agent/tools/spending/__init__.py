"""The _build_filters-based cluster: get_spending_by_category,
get_spending_over_time, search_awards, search_subawards,
get_spending_by_geography. All five funnel their filter params through the
shared tool_filters._build_filters. Each tool lives in its own module
(category.py, over_time.py, search.py, geography.py); this file re-exports
their public names so existing import sites don't need to change.
"""
from __future__ import annotations

from .category import (
    VALID_CATEGORIES,
    Category,
    _normalize_category,
    get_spending_by_category,
    get_spending_by_category_raw,
)
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
)

__all__ = [
    "VALID_CATEGORIES",
    "VALID_GROUPS",
    "Category",
    "Group",
    "_format_geography_result",
    "_normalize_category",
    "_normalize_group",
    "get_spending_by_category",
    "get_spending_by_category_raw",
    "get_spending_by_geography",
    "get_spending_by_geography_raw",
    "get_spending_over_time",
    "get_spending_over_time_raw",
    "search_awards",
    "search_awards_raw",
    "search_subawards",
    "search_subawards_raw",
]

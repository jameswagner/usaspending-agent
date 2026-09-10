"""resolve_county_fips - resolve a county name to the 3-digit FIPS code
search_awards/get_spending_by_category's performed_in_county/
recipient_in_county filters want. See #8: unlike county name itself
(which "Jefferson County, AL" style names work for), Louisiana parishes/
Alaska boroughs/census areas need their local suffix stripped before the
live autocomplete/location endpoint returns a county match at all -
confirmed live "Orleans Parish" returns zero county matches while bare
"Orleans" returns the correct one.
"""
from __future__ import annotations

import re

from anthropic import beta_tool

from ..singletons import _get_usaspending_client
from ..tool_filters import _normalize_county_fips
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted

# Local county-equivalent terminology that breaks the live endpoint's county
# match when included - confirmed live for "Parish" (LA); "Borough" and
# "Census Area" (AK) are the same shape of gap per the Census FIPS scheme,
# not individually live-tested.
_COUNTY_EQUIVALENT_SUFFIX_RE = re.compile(r"\s+(Parish|Borough|Census Area)$", re.IGNORECASE)


def _query_candidates(description: str) -> list[str]:
    """Progressively simplified queries to try, stopping at the first that
    returns a county match. A trailing ", <state>" breaks the match
    entirely (confirmed live for both "Yavapai County, AZ" and "Orleans
    Parish, LA" - zero results, not just a bad ranking), independently of
    the Parish/Borough/Census Area problem - so the comma is stripped
    first, then the local-suffix stripping is tried on what's left.
    """
    candidates = [description]
    before_comma = description.split(",", 1)[0].strip()
    if before_comma != description:
        candidates.append(before_comma)
    for c in list(candidates):
        stripped = _COUNTY_EQUIVALENT_SUFFIX_RE.sub("", c)
        if stripped != c:
            candidates.append(stripped)
    return candidates


@beta_tool
def resolve_county_fips(description: str) -> str:
    """Find the county FIPS code matching a county name, for the performed_in_county/recipient_in_county filters on search_awards/get_spending_by_category/get_spending_over_time/get_spending_by_geography. Always pair the result with the matching state - a county name/code alone is ambiguous across states.

    Args:
        description: A county name, e.g. "Yavapai County", "Jefferson Parish", or just "Orleans".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    client = _get_usaspending_client()

    counties = []
    for candidate in _query_candidates(description):
        response = client.autocomplete_location(candidate)
        counties = response.results.counties
        if counties:
            break

    if not counties:
        return f"No county found matching '{description}'."

    _record_tool_call("resolve_county_fips", counties, {"description": description})

    lines = [
        f"{c.county_name} County, {c.state_name} - FIPS {_normalize_county_fips(c.county_fips)} "
        f"(pair with state={c.state_name})"
        for c in counties
    ]
    return _wrap_untrusted("\n".join(lines))

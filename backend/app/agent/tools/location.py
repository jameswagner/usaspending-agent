"""resolve_county_fips - resolve a county name to the 3-digit FIPS code
search_awards/get_spending_by_category's performed_in_county/
recipient_in_county filters want. Louisiana parishes/Alaska boroughs/
census areas need their local suffix stripped before the live
autocomplete/location endpoint returns a county match at all - confirmed
live "Orleans Parish" returns zero county matches while bare "Orleans"
returns the correct one.

The live endpoint also hard-caps county results at 10 with no
state-scoping param, so a common name can push the real match out of the
response entirely. A bundled Census county reference table fills that gap.
"""
from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from anthropic import beta_tool

from backend.app.usaspending import USASpendingAPIError

from ..singletons import _get_usaspending_client
from ..tool_filters import US_STATE_ABBREVIATIONS, _normalize_county_fips
from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted

_COUNTY_EQUIVALENT_SUFFIX_RE = re.compile(r"\s+(Parish|Borough|Census Area)$", re.IGNORECASE)

_GENERIC_COUNTY_SUFFIX_RE = re.compile(
    r"\s+(County|Parish|Borough|Census Area|Municipality|Municipio|City and Borough)$", re.IGNORECASE
)

_COUNTY_DATA_PATH = Path(__file__).parent / "data" / "us_counties.txt"
_STATE_NAME_BY_ABBR = {abbr: name.upper() for name, abbr in US_STATE_ABBREVIATIONS.items()}


def _bare_county_name(name: str) -> str:
    return _GENERIC_COUNTY_SUFFIX_RE.sub("", name).strip().lower()


@lru_cache(maxsize=1)
def _load_county_reference() -> list[tuple[str, str, str]]:
    # www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt, trimmed to 3 columns.
    rows = []
    with _COUNTY_DATA_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            rows.append((row["STATE"], row["COUNTYFP"], row["COUNTYNAME"]))
    return rows


def _match_static_counties(description: str) -> list[tuple[str, str, str]]:
    bare_candidates = {_bare_county_name(c) for c in _query_candidates(description)}
    return [row for row in _load_county_reference() if _bare_county_name(row[2]) in bare_candidates]


def _live_county_key(county) -> tuple[str, str]:
    abbr = US_STATE_ABBREVIATIONS.get(county.state_name.strip().lower(), county.state_name.strip().upper())
    return (abbr, _normalize_county_fips(county.county_fips))


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
    try:
        for candidate in _query_candidates(description):
            response = client.autocomplete_location(candidate)
            counties = response.results.counties
            if counties:
                break
    except USASpendingAPIError as e:
        return f"This query failed: {e}."

    lines = [
        f"{c.county_name} County, {c.state_name} - FIPS {_normalize_county_fips(c.county_fips)} "
        f"(pair with state={c.state_name})"
        for c in counties
    ]
    seen = {_live_county_key(c) for c in counties}

    for state_abbr, fips, name in _match_static_counties(description):
        if (state_abbr, fips) in seen:
            continue
        seen.add((state_abbr, fips))
        state_name = _STATE_NAME_BY_ABBR.get(state_abbr, state_abbr)
        lines.append(f"{name}, {state_name} - FIPS {fips} (pair with state={state_name})")

    if not lines:
        return f"No county found matching '{description}'."

    _record_tool_call("resolve_county_fips", counties, {"description": description})

    return _wrap_untrusted("\n".join(lines))

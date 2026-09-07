"""Opt-in coverage check: does the live USASpending API's AdvancedFilterObject
have fields ADVANCED_FILTER_FIELD_COVERAGE (usaspending_client.py) doesn't
know about yet?

Not a monitoring service - a rerunnable version of the manual process the
original audit did by hand (fetch the live contract from GitHub, read the
field list, compare against what this codebase models/exposes). Run before
starting new filter/tool work, not on a schedule:
    uv run python -m backend.app.agent.dev_tools.check_filter_coverage

Fetches spending_by_award.md specifically - confirmed during the audit
(2026-09-06) to have the fullest AdvancedFilterObject definition of the
three endpoint contracts (it's the only one with program_activities/
object_classes/award_unique_id). A field present only in an older-vintage
doc (object_class/program_activity - see AdvancedFilters's docstring for
why both name variants are already tracked) won't show up as "removed"
here since this only ever reports genuinely new/unknown fields, not
absence.

Regex-based extraction of a Markdown API Blueprint doc, not a real parser
- if GitHub reformats the contract, this may need a quick manual look
rather than trusting its output blindly, the same caveat every other
scraping-based dev_tools script in this project carries.
"""
from __future__ import annotations

import re

import requests

from backend.app.usaspending_client import ADVANCED_FILTER_FIELD_COVERAGE

CONTRACT_URL = (
    "https://raw.githubusercontent.com/fedspendingtransparency/usaspending-api/"
    "master/usaspending_api/api_contracts/contracts/v2/search/spending_by_award.md"
)

# Matches a top-level field bullet like:
#   + `award_amounts` (optional, array[AwardAmounts], fixed-type)
# Only scanned within the AdvancedFilterObject section (see main()) so
# nested-object fields (e.g. AwardAmounts' own lower_bound/upper_bound)
# aren't mistaken for top-level filter fields.
_FIELD_LINE = re.compile(r"^\+\s*`([a-z_]+)`")


def _extract_advanced_filter_object_fields(contract_text: str) -> set[str]:
    section_start = contract_text.find("### AdvancedFilterObject (object)")
    if section_start == -1:
        raise ValueError(
            "Couldn't find '### AdvancedFilterObject (object)' in the fetched contract - "
            "the doc's structure may have changed; read it manually before trusting this script."
        )
    next_section = contract_text.find("\n### ", section_start + 1)
    section = contract_text[section_start : next_section if next_section != -1 else None]

    return {m.group(1) for line in section.splitlines() if (m := _FIELD_LINE.match(line))}


def main() -> None:
    print(f"Fetching {CONTRACT_URL} ...")
    resp = requests.get(CONTRACT_URL, timeout=30)
    resp.raise_for_status()

    live_fields = _extract_advanced_filter_object_fields(resp.text)
    known_fields = set(ADVANCED_FILTER_FIELD_COVERAGE.keys())

    new_fields = live_fields - known_fields
    print(f"\nLive AdvancedFilterObject fields found: {len(live_fields)}")
    print(f"Fields already tracked in ADVANCED_FILTER_FIELD_COVERAGE: {len(known_fields)}")

    if new_fields:
        print(f"\nNEW fields the live API has that this codebase doesn't track yet ({len(new_fields)}):")
        for field in sorted(new_fields):
            print(f"  - {field}")
        print(
            "\nThese aren't automatically wrong to leave unmodeled - see "
            "ADVANCED_FILTER_FIELD_COVERAGE's docstring for the graduation rule "
            "(real observed demand, e.g. via LangSmith traces of failed tool calls, "
            "not mere existence in the API)."
        )
    else:
        print("\nNo new fields found - ADVANCED_FILTER_FIELD_COVERAGE is up to date with the live contract.")

    stale_entries = known_fields - live_fields
    if stale_entries:
        print(
            f"\nFor reference, {len(stale_entries)} tracked field(s) weren't found in this specific "
            f"contract file (expected for object_class/program_activity - the older-vintage names, "
            f"see AdvancedFilters's docstring - not necessarily a real removal):"
        )
        for field in sorted(stale_entries):
            print(f"  - {field}")


if __name__ == "__main__":
    main()

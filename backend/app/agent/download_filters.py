"""Shared vocabulary for mapping a spending tool's own resolved call
arguments/context into the download pilot's filter field names
(DownloadIntent in download_handler.py).

Split into its own module, with no dependency on download_handler.py or
response_shaping.py, so both can import this table without creating a
circular import between them: download_handler.py needs it for the text
download path's follow-up carryover (_extract_prior_tool_context), and
response_shaping.py needs the identical mapping for follow_ups_for's
"Download this" button, which has to build the same download-shaped filter
dict from a tool call's own recorded context.
"""
from __future__ import annotations

# Maps a spending tool's own argument/context key names to the DownloadIntent
# field they correspond to - only fields DownloadIntent understands.
# Deliberately excludes display-only args like `limit`/`category`/`sort_by`/
# `group`/`geo_layer`/`geo_layer_filters`/`scope` (get_spending_by_geography's
# grouping axis) - a breakdown's top-N, sort order, or grouping dimension is
# not a scope filter and must never be carried into a download.
#
# subrecipient_name/subrecipient_in_* (search_subawards's own arg names) map
# onto the same recipient_name/recipient_in_* download fields - confirmed in
# search_subawards_raw that they're passed into _build_filters's
# recipient_name/recipient_in_* parameters directly, same underlying filter.
#
# recipient_id is deliberately absent - see DOWNLOAD_UNSUPPORTED_SCOPE_ARGS.
TOOL_ARG_TO_DOWNLOAD_FIELD = {
    "agency_name": "agency_raw",
    "award_type": "award_type",
    "start_year": "start_year",
    "end_year": "end_year",
    "time_period_type": "time_period_type",
    "recipient_name": "recipient_name",
    "subrecipient_name": "recipient_name",
    "min_amount": "min_amount",
    "max_amount": "max_amount",
    "performed_in_state": "performed_in_state",
    "recipient_in_state": "recipient_in_state",
    "subrecipient_in_state": "recipient_in_state",
    "performed_in_county": "performed_in_county",
    "recipient_in_county": "recipient_in_county",
    "subrecipient_in_county": "recipient_in_county",
    "performed_in_city": "performed_in_city",
    "recipient_in_city": "recipient_in_city",
    "subrecipient_in_city": "recipient_in_city",
    "performed_in_zip": "performed_in_zip",
    "recipient_in_zip": "recipient_in_zip",
    "subrecipient_in_zip": "recipient_in_zip",
    "performed_in_district": "performed_in_district",
    "recipient_in_district": "recipient_in_district",
    "subrecipient_in_district": "recipient_in_district",
    "keywords": "keywords",
    "date_type": "date_type",
    "place_of_performance_scope": "place_of_performance_scope",
    "recipient_scope": "recipient_scope",
    "naics_code": "naics_code",
    "psc_code": "psc_code",
    "cfda_program": "cfda_program",
    "award_id": "award_id",
    "recipient_type": "recipient_type",
    "description": "description",
    "tas_code": "tas_code",
    "federal_account": "federal_account",
    "def_codes": "def_codes",
    "contract_pricing_type": "contract_pricing_type",
    "set_aside_type": "set_aside_type",
    "extent_competed_type": "extent_competed_type",
}

# Real scoping filters a spending tool can resolve that /api/v2/download/search/'s
# own Filters object has no field for. Live-verified 2026-09-30: posting a job with
# a bogus recipient_id alongside a real agency+time_period scope produced the exact
# same file_name/job as the identical request with recipient_id omitted entirely -
# the field is silently dropped before the query ever runs, not merely ignored
# server-side after being recorded. Carrying it forward would silently widen the
# download past what the answer it continues was scoped to, so it's excluded from
# TOOL_ARG_TO_DOWNLOAD_FIELD above and instead surfaced as a caveat (download_handler.py)
# or used to suppress the follow-up button entirely when it's the only scope available
# (response_shaping.py's follow_ups_for).
DOWNLOAD_UNSUPPORTED_SCOPE_ARGS = {
    "recipient_id": "the recipient ID filter",
}

# Every DownloadIntent field _build_filters can actually consume, beyond
# agency_raw/time_period_type/start_year/end_year/award_type/spending_level,
# which both download_handler.py and response_shaping.py handle by hand since
# they're resolved slightly differently (e.g. agency_raw needs a lookup;
# spending_level has a per-caller default). One list so every caller that
# needs "every carryover-able filter field" stays in sync with DownloadIntent.
CARRYOVER_FILTER_FIELDS = [
    "recipient_name", "min_amount", "max_amount",
    "performed_in_state", "recipient_in_state",
    "performed_in_county", "recipient_in_county",
    "performed_in_city", "recipient_in_city",
    "performed_in_zip", "recipient_in_zip",
    "performed_in_district", "recipient_in_district",
    "keywords", "date_type", "place_of_performance_scope", "recipient_scope",
    "naics_code", "psc_code", "cfda_program", "award_id", "recipient_type",
    "description", "tas_code", "federal_account", "def_codes",
    "contract_pricing_type", "set_aside_type", "extent_competed_type",
]

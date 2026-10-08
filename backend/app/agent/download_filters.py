"""Tool-arg to DownloadIntent field mapping, split out so download_handler and response_shaping can share it without a circular import."""
from __future__ import annotations

# Display-only args (limit/category/sort_by/group/geo_layer/scope) are excluded: they aren't scope filters.
# subrecipient_* args map onto the same recipient_* download fields.
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

# Scope filters /download/search/ silently drops; surfaced as a caveat, or suppresses the follow-up button when it's the only scope.
DOWNLOAD_UNSUPPORTED_SCOPE_ARGS = {
    "recipient_id": "the recipient ID filter",
}

# DownloadIntent fields _build_filters consumes, beyond the core agency/time/award_type/spending_level handled by hand.
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

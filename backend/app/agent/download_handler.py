"""Download pipeline shared by the download tools and the "Download this" button, backed by POST /api/v2/download/search/."""
from __future__ import annotations

import json
import logging
import time
from typing import Literal

from pydantic import BaseModel

from backend.app.usaspending import (
    BASE_URL,
    AdvancedFilters,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

from .download_filters import (
    CARRYOVER_FILTER_FIELDS,
)
from .response_shaping import (
    DownloadSpec,
    ToolCitation,
    year_label,
)
from .singletons import _get_usaspending_client
from .tool_filters import (
    _BROAD_CATEGORY_FOR_AWARD_TYPE,
    _build_filters,
    _normalize_award_type,
    _pop_naics_disclosure,
)

logger = logging.getLogger(__name__)


_POLL_INTERVAL_SECONDS = 4
_POLL_TIMEOUT_SECONDS = 90

# Fixed - this module rules out free-form column selection. award_id_piid/award_id_fain
# are deliberately NOT hard-coded into "awards"/"transactions" here - see
# _identifier_columns_for_award_type below for why, and _DOWNLOAD_COLUMNS_BASE_BY_LEVEL
# for the rest of each level's fixed columns.
_DOWNLOAD_COLUMNS_BASE_BY_LEVEL = {
    "awards": [
        "recipient_name",
        "total_obligated_amount",
        "period_of_performance_start_date",
        "awarding_agency_name",
    ],
    # Verified live 2026-09-24 - transactions use per-action field names (federal_action_obligation,
    # action_date), not the award-level totals/dates above, which don't apply to a single transaction.
    "transactions": [
        "recipient_name",
        "federal_action_obligation",
        "action_date",
        "awarding_agency_name",
    ],
    # Verified live 2026-09-24 - subawards use their own prime_award_*/subaward*/subawardee_* prefix,
    # not the prime-award column names above (e.g. award_id_piid isn't valid here, prime_award_piid is;
    # a job posted with the wrong names for this level is accepted at request time then fails async).
    # Not affected by the piid/fain split below - prime_award_piid is its own fixed column here,
    # not swapped per award_type.
    "subawards": [
        "prime_award_piid",
        "subawardee_name",
        "subaward_amount",
        "subaward_action_date",
        "prime_award_awarding_agency_name",
    ],
}

# AWARD_TYPE_GROUPS leaf categories (tool_filters.py's _BROAD_CATEGORY_FOR_AWARD_TYPE)
# that are procurement, not assistance - the only families with a PIID at all.
_PROCUREMENT_AWARD_CATEGORIES = {"contracts", "idv"}


def _identifier_columns_for_award_type(award_type: str | None) -> list[str]:
    """Picks award_id_piid vs award_id_fain for an "awards"/"transactions" download.

    Live-verified 2026-10-01: requesting the identifier that's 100% null for the
    narrowed award_type (fain for contracts, piid for grants) crashes
    /api/v2/download/search/ outright ("An error occurred.") - not just wasted, fatal.
    Only contracts/idv are PIID-bearing; the other six AWARD_TYPE_GROUPS leaf
    categories are assistance awards, identified by FAIN.
    """
    if award_type is None:
        return ["award_id_piid", "award_id_fain"]
    category = _BROAD_CATEGORY_FOR_AWARD_TYPE[_normalize_award_type(award_type)]
    return ["award_id_piid"] if category in _PROCUREMENT_AWARD_CATEGORIES else ["award_id_fain"]


# Download-side column for naics_code/psc_code when either is a real scope filter
# (as opposed to a category breakdown - see _DOWNLOAD_COLUMN_BY_CATEGORY). The search
# endpoint's display names ("NAICS"/"PSC") are NOT valid here - same crash as above.
_CATEGORY_COLUMN_BY_FILTER_FIELD = {
    "naics_code": "naics_code",
    "psc_code": "product_or_service_code",
}

# get_spending_by_category's `category` values, mapped to the download-side column
# that shows it per row - from fedspendingtransparency/usaspending-api's own
# download_column_historical_lookups.py, live-verified 2026-10-01 against
# /api/v2/download/search/ directly (that file predates the ES-backed endpoint and
# one entry, disaster_emergency_fund_codes, didn't carry over). awarding_agency/
# recipient need no entry - already in _DOWNLOAD_COLUMNS_BASE_BY_LEVEL. district/defc
# are deliberately absent - no working column name found, not guessed around.
_DOWNLOAD_COLUMN_BY_CATEGORY = {
    "naics": "naics_code",
    "psc": "product_or_service_code",
    "cfda": "cfda_number",
    "country": "primary_place_of_performance_country_code",
    "county": "primary_place_of_performance_county_name",
    "state_territory": "primary_place_of_performance_state_code",
    "awarding_subagency": "awarding_sub_agency_name",
    "funding_agency": "funding_agency_name",
    "funding_subagency": "funding_sub_agency_name",
    "recipient_duns": "recipient_duns",
    # Live-verified but notably slower (~100s vs. single-digit seconds for the others) -
    # this one resolves via a subquery annotation, not a plain indexed field.
    "federal_account": "federal_accounts_funding_this_award",
}


def _category_columns_for(naics_code: str | None, psc_code: str | None, category: str | None) -> list[str]:
    """Columns showing which category a row belongs to - triggered by either a real
    naics_code/psc_code scope filter, or get_spending_by_category's own `category`
    grouping dimension (see _DOWNLOAD_COLUMN_BY_CATEGORY). Deduplicated so a call
    scoped by naics_code AND broken down by category="naics" doesn't request the
    same column twice."""
    columns = []
    if naics_code is not None:
        columns.append(_CATEGORY_COLUMN_BY_FILTER_FIELD["naics_code"])
    if psc_code is not None:
        columns.append(_CATEGORY_COLUMN_BY_FILTER_FIELD["psc_code"])
    category_column = _DOWNLOAD_COLUMN_BY_CATEGORY.get(category)
    if category_column is not None and category_column not in columns:
        columns.append(category_column)
    return columns


def _download_columns_for(
    spending_level: str,
    award_type: str | None,
    naics_code: str | None = None,
    psc_code: str | None = None,
    category: str | None = None,
) -> list[str]:
    base = _DOWNLOAD_COLUMNS_BASE_BY_LEVEL[spending_level]
    if spending_level == "subawards":
        return base
    return (
        _identifier_columns_for_award_type(award_type)
        + _category_columns_for(naics_code, psc_code, category)
        + base
    )

# Verified live 2026-09-24: /download/search/'s spending_level array members are fully
# independent (["awards"] alone excludes sub-awards) - unlike the legacy /download/awards/
# and /download/transactions/ endpoints, which always bundle subawards in. This table
# preserves that legacy bundling so switching to /download/search/ is a strict widening,
# not a silent regression for the two levels already shipped before this table existed.
_SPENDING_LEVEL_TO_API_ARRAY = {
    "awards": ["awards", "subawards"],
    "transactions": ["transactions", "subawards"],
    "subawards": ["subawards"],
}


_SINGLE_AWARD_ENDPOINT_BY_PREFIX = (
    ("CONT_AWD_", "contract"),
    ("CONT_IDV_", "idv"),
    ("ASST_", "assistance"),
)
_SINGLE_AWARD_CLIENT_METHOD = {
    "contract": "download_contract",
    "assistance": "download_assistance",
    "idv": "download_idv",
}


def _award_id_endpoint(award_id: str) -> str | None:
    for prefix, endpoint in _SINGLE_AWARD_ENDPOINT_BY_PREFIX:
        if award_id.startswith(prefix):
            return endpoint
    return None


class DownloadIntent(BaseModel):
    agency_raw: str | None = None
    time_period_type: Literal["fiscal", "calendar"] = "fiscal"
    start_year: int | None = None
    end_year: int | None = None
    # Set together instead of start_year/end_year for a period narrower than a full year.
    start_date: str | None = None
    end_date: str | None = None
    award_type: str | None = None
    spending_level: Literal["awards", "transactions", "subawards"] = "awards"
    # Mirrors SpendingFilterParams except recipient_id (see download_filters.DOWNLOAD_UNSUPPORTED_SCOPE_ARGS).
    recipient_name: str | None = None
    min_amount: float | None = None
    max_amount: float | None = None
    performed_in_state: str | None = None
    recipient_in_state: str | None = None
    performed_in_county: str | None = None
    recipient_in_county: str | None = None
    performed_in_city: str | None = None
    recipient_in_city: str | None = None
    performed_in_zip: str | None = None
    recipient_in_zip: str | None = None
    performed_in_district: str | None = None
    recipient_in_district: str | None = None
    keywords: str | None = None
    date_type: str | None = None
    place_of_performance_scope: str | None = None
    recipient_scope: str | None = None
    naics_code: str | None = None
    psc_code: str | None = None
    cfda_program: str | None = None
    award_id: str | None = None
    recipient_type: str | None = None
    description: str | None = None
    tas_code: str | None = None
    federal_account: str | None = None
    def_codes: list[str] | None = None
    contract_pricing_type: list[str] | None = None
    set_aside_type: list[str] | None = None
    extent_competed_type: list[str] | None = None
    # NOT a real scope filter (never passed to _build_filters) - carries
    # get_spending_by_category's own grouping dimension for _category_columns_for.
    category: str | None = None


def _get_status_tolerating_early_404(client: USASpendingClient, file_name: str):
    """A 404 here means the job record isn't indexed yet, not that it doesn't exist - confirmed live."""
    try:
        return client.get_download_status(file_name)
    except USASpendingAPIError as e:
        if str(e).startswith("404"):
            return None
        raise


def _poll_until_finished(client: USASpendingClient, file_name: str):
    """Status moves ready -> running -> finished/failed; both ready and running mean "keep polling"."""
    elapsed = 0.0
    status = _get_status_tolerating_early_404(client, file_name)
    while (status is None or status.status in ("ready", "running")) and elapsed < _POLL_TIMEOUT_SECONDS:
        time.sleep(_POLL_INTERVAL_SECONDS)
        elapsed += _POLL_INTERVAL_SECONDS
        status = _get_status_tolerating_early_404(client, file_name)
    if status is None:
        raise USASpendingAPIError(
            f"Download job {file_name} never became queryable within {_POLL_TIMEOUT_SECONDS}s."
        )
    return status


def _build_download_citation(
    agency_name: str | None, intent: DownloadIntent, filters: AdvancedFilters, columns: list[str], api_spending_level: list[str]
) -> ToolCitation:
    """Built locally, not via the shared capture-buffer drain - that contextvar set doesn't reliably propagate out of @traceable."""
    params: dict[str, str | int | float] = {
        "agency_name": agency_name or "all agencies",
        "spending_level": intent.spending_level,
    }
    if intent.start_date and intent.end_date:
        params["start_date"] = intent.start_date
        params["end_date"] = intent.end_date
        period_label = f"{intent.start_date} to {intent.end_date}"
    else:
        params["start_year"] = intent.start_year
        params["end_year"] = intent.end_year
        period_label = f"{year_label(intent.time_period_type, intent.start_year)}-{year_label(intent.time_period_type, intent.end_year)}"
    if intent.award_type:
        params["award_type"] = intent.award_type
    # Display only - the curl body below holds the exact real request. ToolCitation.parameters
    # has no list type, so a list-valued filter (def_codes/contract_pricing_type/etc.) is
    # comma-joined here the same lossy-for-display-only way _merge_optional_filter_params does
    # for the other spending tools' citations.
    for field in CARRYOVER_FILTER_FIELDS:
        value = getattr(intent, field)
        if value is None:
            continue
        params[field] = ", ".join(value) if isinstance(value, list) else value
    description = f"{intent.spending_level.capitalize()} CSV download, {agency_name or 'all agencies'}, {period_label}"
    body = {
        "filters": filters.model_dump(exclude_none=True),
        "columns": columns,
        "file_format": "csv",
        "spending_level": api_spending_level,
    }
    curl = f"curl -X POST '{BASE_URL}/api/v2/download/search/' -H 'Content-Type: application/json' -d '{json.dumps(body)}'"
    return ToolCitation(tool_name="download_search", parameters=params, description=description, curl=curl)


def _execute_download(intent: DownloadIntent, agency_name: str | None, conversation_id: str):
    from .orchestrator import AgentResult

    client = _get_usaspending_client()
    carryover_filters = {field: getattr(intent, field) for field in CARRYOVER_FILTER_FIELDS}

    try:
        # Placeholder years when start_date is set - real scope is applied below by overwriting time_period.
        start_year = int(intent.start_date[:4]) if intent.start_date else intent.start_year
        end_year = start_year if intent.start_date else intent.end_year
        filters = _build_filters(
            client,
            agency_name,
            intent.time_period_type,
            start_year,
            end_year,
            award_type=intent.award_type,
            scope_required=False,
            **carryover_filters,
        )
        naics_note = _pop_naics_disclosure()
        if intent.start_date and intent.end_date:
            filters.time_period = [TimePeriod(start_date=intent.start_date, end_date=intent.end_date)]
        columns = _download_columns_for(
            intent.spending_level, intent.award_type, intent.naics_code, intent.psc_code, intent.category
        )
        api_spending_level = _SPENDING_LEVEL_TO_API_ARRAY[intent.spending_level]
        job = client.download_search(filters, columns, api_spending_level)
        citation = _build_download_citation(agency_name, intent, filters, columns, api_spending_level)
        status = _poll_until_finished(client, job.file_name)
    except USASpendingAPIError as e:
        logger.warning("Download pipeline failed: %s", e)
        return AgentResult(
            answer_text=f"This download failed: {e}.",
            conversation_id=conversation_id,
        )

    download = DownloadSpec(
        file_name=status.file_name, url=status.file_url, status_url=job.status_url,
        status=status.status, total_rows=status.total_rows,
    )
    if status.status == "finished":
        answer_text = (
            f"Your download is ready: {status.total_rows or 0} rows in {status.file_name}.\n{status.file_url}"
        )
    elif status.status == "failed":
        answer_text = f"This download failed to generate: {status.message or 'no further detail from the API.'}"
    else:
        answer_text = (
            f"Your download is still generating ({status.file_name}). It'll appear at the link below "
            "once ready - check back shortly."
        )
    if naics_note:
        answer_text += f"\n\nNote: {naics_note}"

    return AgentResult(
        answer_text=answer_text, conversation_id=conversation_id, downloads=[download], tool_citations=[citation]
    )


def handle_download_followup(filters: dict, conversation_id: str):
    """Click path for the "Download this" button: builds the intent straight from structured filters, no NL extraction."""
    from .orchestrator import AgentResult

    intent = DownloadIntent(**filters)
    agency_name = None
    if intent.agency_raw:
        client = _get_usaspending_client()
        match = client.find_agency_by_name(intent.agency_raw)
        if match is None:
            return AgentResult(
                answer_text=f"No agency found matching '{intent.agency_raw}'.",
                conversation_id=conversation_id,
            )
        agency_name = match.agency_name

    return _execute_download(intent, agency_name, conversation_id)

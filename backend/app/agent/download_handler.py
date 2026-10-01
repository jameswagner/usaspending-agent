"""Deterministic pre-tool-loop download-request pipeline, backed by POST /api/v2/download/search/."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Literal

from pydantic import BaseModel, ValidationError

from backend.app.usaspending import (
    BASE_URL,
    AdvancedFilters,
    DownloadJobResponse,
    TimePeriod,
    USASpendingAPIError,
    USASpendingClient,
)

from .response_shaping import (
    DownloadSpec,
    ToolCitation,
    current_fiscal_year,
    year_label,
)
from .scope import _render_recent_exchanges
from .singletons import MODEL, _get_client, _get_usaspending_client
from .tool_filters import (
    _BROAD_CATEGORY_FOR_AWARD_TYPE,
    _build_filters,
    _normalize_award_type,
    _pop_naics_disclosure,
)

logger = logging.getLogger(__name__)

_DOWNLOAD_INTENT_PATTERN = (
    "download", "export", "csv", "spreadsheet", "give me the data", "give me a file",
    "raw data", "data file",
)

# Endpoints this module defers - matched before the extraction call runs, so an unsupported request never burns one.
# "transaction"/"sub-award"/"subaward" are deliberately absent - spending_level (below) now covers them.
_UNSUPPORTED_DOWNLOAD_PATTERN = {
    "account": "account-level data",
    "idv": "IDV data",
    "indefinite delivery": "IDV data",
    "disaster": "disaster/relief-specific data",
}

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
    """Which of award_id_piid/award_id_fain to request for an "awards"/"transactions"
    download, given the resolved award_type (or None if unscoped by type).

    Live-verified 2026-10-01: requesting the identifier column that's 100% null across
    every row of a narrowed award_type crashes /api/v2/download/search/ outright
    ("An error occurred.", no further detail - every exception here is masked server-
    side, see fedspendingtransparency/usaspending-api#4690) - not merely wasted, but
    fatal. Confirmed in both directions: award_id_fain (always null for a contracts-
    only request) and award_id_piid (always null for a grants-only request) each
    reliably crash the job; the correct column alone works every time, and so does
    requesting both when award_type is unset and the result set spans multiple award
    families (neither column is 100% null across a mixed set).

    Only two of AWARD_TYPE_GROUPS' eight leaf categories are procurement (PIID-bearing):
    contracts and idv. The other six - grants, loans, insurance,
    other_financial_assistance, direct_payment_specified, direct_payment_unrestricted -
    are all assistance awards, identified by FAIN instead.
    """
    if award_type is None:
        return ["award_id_piid", "award_id_fain"]
    category = _BROAD_CATEGORY_FOR_AWARD_TYPE[_normalize_award_type(award_type)]
    return ["award_id_piid"] if category in _PROCUREMENT_AWARD_CATEGORIES else ["award_id_fain"]


# Which download-side column shows naics_code/psc_code when either is a real scope
# filter (as opposed to get_spending_by_category's own `category` grouping dimension -
# see _DOWNLOAD_COLUMN_BY_CATEGORY below for that). The *search* endpoint's own display
# field names (e.g. "NAICS"/"PSC" from spending_by_award.md) are NOT valid here -
# confirmed live 2026-10-01 that both crash /api/v2/download/search/ the same "accepted
# at request time, fails async" way an always-null identifier column does (see
# _identifier_columns_for_award_type above); the download endpoint's `columns` takes
# the underlying search-index field name instead.
_CATEGORY_COLUMN_BY_FILTER_FIELD = {
    "naics_code": "naics_code",
    "psc_code": "product_or_service_code",
}

# get_spending_by_category's own `category` grouping-dimension values, mapped to the
# live-verified download-side column that shows it per row - useful even when the
# breakdown wasn't filtered to one specific value (a NAICS breakdown ranking across
# every code has no naics_code scope value to key off, but each row still has its own
# real NAICS code worth showing in the download). Derived from
# fedspendingtransparency/usaspending-api's own download_column_historical_lookups.py,
# then independently confirmed live against /api/v2/download/search/ 2026-10-01 - not
# assumed from the source alone, since that file predates the Elasticsearch-backed
# /download/search/ endpoint and at least one of its entries (disaster_emergency_fund_
# codes, below) turned out not to carry over.
#
# awarding_agency/recipient need no entry - awarding_agency_name/recipient_name are
# already in _DOWNLOAD_COLUMNS_BASE_BY_LEVEL for every "awards"/"transactions" download.
#
# district and defc are deliberately absent, not guessed around: district has no
# corresponding column anywhere in that lookup file at all, and defc's own field
# (disaster_emergency_fund_codes) crashed live both plain and with the lookup file's
# own NAMING_CONFLICT_DISCRIMINATOR suffix - genuinely unresolved, left for a future
# fix once a working column name is actually found.
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


def _looks_like_download_request(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _DOWNLOAD_INTENT_PATTERN)


def _is_download_followup(recent_messages: list | None) -> bool:
    """Keyed on the intent _persist_download_turn stashes, not on the answer's wording -
    prefix-matching user-facing copy meant rewording a message silently broke detection."""
    for message in reversed(recent_messages or []):
        if getattr(message, "type", None) == "ai" and isinstance(message.content, str):
            return bool((getattr(message, "additional_kwargs", None) or {}).get("download_intent"))
    return False


def _unsupported_download_label(question: str) -> str | None:
    q = question.lower()
    for term, label in _UNSUPPORTED_DOWNLOAD_PATTERN.items():
        if term in q:
            return label
    return None


# generated_unique_award_id's prefixes are fully disjoint - startswith mapping is unambiguous.
_SINGLE_AWARD_ID_PATTERN = re.compile(r"\b(?:CONT_AWD_|CONT_IDV_|ASST_)[A-Za-z0-9_\-]+")
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

# Only these phrasings are worth a history search - not every download request.
_SINGLE_AWARD_FOLLOWUP_TERMS = (
    "this award", "this contract", "this grant", "this loan", "this idv",
    "that award", "that contract", "that grant", "that loan", "that idv",
    "the same award", "download it", "download that",
)

# Matches the "[internal_id: ...]" tag search_awards/get_award_details append to each result.
_PRIOR_INTERNAL_ID_PATTERN = re.compile(r"\[internal_id:\s*([A-Za-z0-9_\-]+)\]")


def _award_id_endpoint(award_id: str) -> str | None:
    for prefix, endpoint in _SINGLE_AWARD_ENDPOINT_BY_PREFIX:
        if award_id.startswith(prefix):
            return endpoint
    return None


def _extract_award_id_from_history(recent_messages: list | None) -> str | None:
    for message in reversed(recent_messages or []):
        if getattr(message, "type", None) != "tool" or not isinstance(message.content, str):
            continue
        match = _PRIOR_INTERNAL_ID_PATTERN.search(message.content)
        if match:
            return match.group(1)
    return None


def _is_single_award_followup_phrase(question: str) -> bool:
    q = question.lower()
    return any(term in q for term in _SINGLE_AWARD_FOLLOWUP_TERMS)


def _resolve_single_award_id(question: str, recent_messages: list | None) -> str | None:
    match = _SINGLE_AWARD_ID_PATTERN.search(question)
    if match:
        return match.group(0)
    if _is_single_award_followup_phrase(question):
        return _extract_award_id_from_history(recent_messages)
    return None


# Maps a spending tool's own argument names to the DownloadIntent field they
# correspond to - only fields DownloadIntent understands. Deliberately excludes
# display-only args like `limit`/`category`/`sort_by`/`group`/`geo_layer`/
# `geo_layer_filters`/`scope` (get_spending_by_geography's grouping axis) - a
# breakdown's top-N, sort order, or grouping dimension is not a scope filter
# and must never be carried into a download.
#
# subrecipient_name/subrecipient_in_* (search_subawards's own arg names) map
# onto the same recipient_name/recipient_in_* download fields - confirmed in
# search_subawards_raw that they're passed into _build_filters's
# recipient_name/recipient_in_* parameters directly, same underlying filter.
#
# recipient_id is deliberately absent - see _DOWNLOAD_UNSUPPORTED_SCOPE_ARGS.
_TOOL_ARG_TO_DOWNLOAD_FIELD = {
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
# _TOOL_ARG_TO_DOWNLOAD_FIELD above and instead surfaced as a caveat.
_DOWNLOAD_UNSUPPORTED_SCOPE_ARGS = {
    "recipient_id": "the recipient ID filter",
}


def _scope_caveat(dropped_filter_labels: list[str], naics_note: str | None = None) -> str:
    """Builds the caveat suffix appended to a download's answer_text when part of the
    prior answer's scope couldn't be carried through - see _DOWNLOAD_UNSUPPORTED_SCOPE_ARGS.
    Returns "" when there's nothing to say, so this is safe to always append."""
    notes = []
    if dropped_filter_labels:
        notes.append(
            f"the prior answer's {', '.join(dropped_filter_labels)} couldn't be carried into this "
            "download - the download endpoint has no equivalent filter. This file may include more "
            "than what you saw."
        )
    if naics_note:
        notes.append(naics_note)
    return "\n\n" + "\n".join(f"Note: {n}" for n in notes) if notes else ""


def _extract_prior_tool_context(
    recent_messages: list | None, max_messages: int = 10
) -> tuple[dict | None, list[str]]:
    """Looks back through the real LangGraph message list - not the lossy prose
    _render_recent_exchanges produces - for the most recently resolved structured
    context: either this pilot's own prior DownloadIntent (stashed by
    _persist_download_turn) or a spending tool's actual resolved call arguments.
    Gives a follow-up extraction real values to copy instead of re-deriving
    everything from text, which left room to invent fields nothing ever stated.

    Also returns the human-readable labels of any real scope filter the prior
    call used that has no download-side equivalent (see
    _DOWNLOAD_UNSUPPORTED_SCOPE_ARGS), so the caller can say so rather than
    silently dropping it."""
    for message in reversed((recent_messages or [])[-max_messages:]):
        if getattr(message, "type", None) != "ai":
            continue
        stashed = (getattr(message, "additional_kwargs", None) or {}).get("download_intent")
        if stashed:
            return dict(stashed), []
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            args = tool_calls[-1].get("args", {})
            context = {
                field: args[arg_name]
                for arg_name, field in _TOOL_ARG_TO_DOWNLOAD_FIELD.items()
                if args.get(arg_name) is not None
            }
            # category (get_spending_by_category's grouping dimension) is NOT a scope
            # filter - never passed to _build_filters (see _CARRYOVER_FILTER_FIELDS,
            # which deliberately excludes it) - but still carried here so a NAICS/PSC
            # breakdown's download can show each row's own category column even though
            # the breakdown itself wasn't filtered to one value. See _category_columns_for.
            if args.get("category") is not None:
                context["category"] = args["category"]
            dropped = [
                label for arg_name, label in _DOWNLOAD_UNSUPPORTED_SCOPE_ARGS.items()
                if args.get(arg_name) is not None
            ]
            return (context or None), dropped
    return None, []


# Every DownloadIntent field _build_filters can actually consume, beyond the
# original agency/time/award_type/spending_level fields _download_intent_context
# already handles by hand above - kept as one list so both that function and the
# _build_filters(**filters) call below stay in sync with DownloadIntent's fields.
_CARRYOVER_FILTER_FIELDS = [
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


def _download_intent_context(intent: DownloadIntent, agency_name: str | None) -> dict:
    """The structured record of what this turn actually resolved - stashed via
    _persist_download_turn so the next follow-up's _extract_prior_tool_context
    has real values instead of prose to work from."""
    context: dict = {"spending_level": intent.spending_level, "time_period_type": intent.time_period_type}
    if agency_name:
        context["agency_raw"] = agency_name
    if intent.start_date and intent.end_date:
        context["start_date"] = intent.start_date
        context["end_date"] = intent.end_date
    else:
        context["start_year"] = intent.start_year
        context["end_year"] = intent.end_year
    if intent.award_type:
        context["award_type"] = intent.award_type
    for field in _CARRYOVER_FILTER_FIELDS:
        value = getattr(intent, field)
        if value is not None:
            context[field] = value
    if intent.category is not None:
        context["category"] = intent.category
    return context


class DownloadIntent(BaseModel):
    # Required in the tool schema below - forces the model to actively decide, not default to True.
    wants_download: bool = True
    agency_raw: str | None = None
    time_period_type: Literal["fiscal", "calendar"] = "fiscal"
    start_year: int | None = None
    end_year: int | None = None
    # Set together instead of start_year/end_year for a period narrower than a full year.
    start_date: str | None = None
    end_date: str | None = None
    award_type: str | None = None
    spending_level: Literal["awards", "transactions", "subawards"] = "awards"
    # Every field below mirrors a SpendingFilterParams field _build_filters already
    # accepts (see tool_filters.py) - widened alongside _TOOL_ARG_TO_DOWNLOAD_FIELD
    # so a download following a scoped answer can actually carry that scope, not
    # just agency/time/award_type. recipient_id is the one SpendingFilterParams
    # field deliberately absent here - see _DOWNLOAD_UNSUPPORTED_SCOPE_ARGS.
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
    # NOT a real scope filter - never passed to _build_filters (see
    # _CARRYOVER_FILTER_FIELDS). Only carries get_spending_by_category's own
    # grouping dimension (e.g. "naics", "psc") across a follow-up, so the
    # resulting download can include that category's column (_category_columns_for)
    # even for a breakdown that wasn't filtered to one specific value.
    category: str | None = None


_EXTRACT_TOOL_NAME = "extract_download_intent"
_EXTRACT_TOOL = {
    "name": _EXTRACT_TOOL_NAME,
    "description": (
        "Extract the agency, time period, and any other scoping filters the user wants a spending "
        "award CSV download for. If the question doesn't name a specific agency or a resolvable "
        "time period, omit those fields rather than guessing - same for every other filter field."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "wants_download": {
                "type": "boolean",
                "description": "True only if this specific question wants an actual exported file, not "
                "just spending information - e.g. a breakdown, comparison, or 'how much' question is "
                "false even if it follows a download in the conversation and even if it names an agency/year.",
            },
            "agency_raw": {"type": "string", "description": "Agency name/abbreviation as stated, or omitted if none is named."},
            "time_period_type": {"type": "string", "enum": ["fiscal", "calendar"]},
            "start_year": {
                "type": "integer",
                "description": "First year of the range. Only use this (with end_year) when the "
                "question asks for a WHOLE fiscal or calendar year - omit it and use start_date/"
                "end_date instead whenever the question names anything narrower (a specific month, "
                "quarter, or date range).",
            },
            "end_year": {"type": "integer", "description": "Last year of the range, paired with start_year."},
            "start_date": {
                "type": "string",
                "description": "YYYY-MM-DD. Use this (with end_date) instead of start_year/end_year "
                "whenever the question names a period narrower than a full year - e.g. 'January "
                "2024' -> start_date='2024-01-01', end_date='2024-01-31'.",
            },
            "end_date": {"type": "string", "description": "YYYY-MM-DD, paired with start_date."},
            "award_type": {
                "type": "string",
                "enum": ["contracts", "grants", "loans"],
                "description": "Omit unless the question specifically names one award type.",
            },
            "spending_level": {
                "type": "string",
                "enum": ["awards", "transactions", "subawards"],
                "description": "awards (default): one row per prime award, its current total value - "
                "use for a plain \"download X's awards/contracts/grants\" request. transactions: one row "
                "per individual transaction/modification - use when the question specifically says "
                "\"transactions\" or \"transaction-level\", not just \"awards\". subawards: one row per "
                "sub-recipient/sub-contractor payment under those prime awards - use when the question "
                "specifically says \"sub-awards\"/\"subcontractors\"/\"subgrantees\", not prime recipients. "
                "Default to awards unless the question clearly names one of the other two.",
            },
            "recipient_name": {"type": "string", "description": "Recipient name text match, e.g. 'Leidos'."},
            "min_amount": {"type": "number", "description": "Lower dollar bound on award amount."},
            "max_amount": {"type": "number", "description": "Upper dollar bound on award amount."},
            "performed_in_state": {"type": "string", "description": "US state/territory where work was performed."},
            "recipient_in_state": {"type": "string", "description": "US state/territory the recipient is located in."},
            "performed_in_county": {"type": "string", "description": "3-digit county FIPS code where work was performed."},
            "recipient_in_county": {"type": "string", "description": "3-digit county FIPS code the recipient is located in."},
            "performed_in_city": {"type": "string", "description": "City where work was performed."},
            "recipient_in_city": {"type": "string", "description": "City the recipient is located in."},
            "performed_in_zip": {"type": "string", "description": "5-digit zip code where work was performed."},
            "recipient_in_zip": {"type": "string", "description": "5-digit zip code the recipient is located in."},
            "performed_in_district": {"type": "string", "description": "Congressional district where work was performed."},
            "recipient_in_district": {"type": "string", "description": "Congressional district the recipient is located in."},
            "keywords": {"type": "string", "description": "Free-text keyword filter."},
            "date_type": {
                "type": "string",
                "enum": ["action_date", "date_signed", "last_modified_date", "new_awards_only"],
                "description": "Which date field the time period filters against, if the question specifies one.",
            },
            "place_of_performance_scope": {"type": "string", "enum": ["domestic", "foreign"]},
            "recipient_scope": {"type": "string", "enum": ["domestic", "foreign"]},
            "naics_code": {"type": "string", "description": "NAICS industry code."},
            "psc_code": {"type": "string", "description": "Product/service code."},
            "cfda_program": {"type": "string", "description": "CFDA/assistance listing number, e.g. '10.001'."},
            "award_id": {"type": "string", "description": "A specific award's PIID/FAIN/URI."},
            "recipient_type": {"type": "string", "description": "Recipient business type, e.g. 'small_business'."},
            "description": {"type": "string", "description": "Phrase to match against the award's own description text."},
            "tas_code": {"type": "string", "description": "A full Treasury Account Symbol code."},
            "federal_account": {"type": "string", "description": "AID-MAIN federal account code, e.g. '028-8704'."},
            "def_codes": {"type": "array", "items": {"type": "string"}, "description": "Disaster Emergency Fund Codes."},
            "contract_pricing_type": {"type": "array", "items": {"type": "string"}, "description": "Contract pricing type keys, e.g. 'firm_fixed_price'."},
            "set_aside_type": {"type": "array", "items": {"type": "string"}, "description": "Contract set-aside type keys."},
            "extent_competed_type": {"type": "array", "items": {"type": "string"}, "description": "Contract extent-competed type keys."},
            "category": {
                "type": "string",
                "description": "Only ever copy this from the previously resolved JSON when continuing a "
                "category breakdown (e.g. a NAICS or PSC breakdown) - never infer or state this from the "
                "new question itself.",
            },
        },
        "required": ["wants_download"],
    },
}


def _extract_download_intent(
    question: str, history_block: str = "", prior_context: dict | None = None
) -> DownloadIntent | None:
    """Returns None on anything that isn't a clean parse, rather than guessing."""
    system = (
        f"Today's fiscal year is FY{current_fiscal_year()}. Use it for relative phrases "
        "like 'this year' or 'most recent year' if the question doesn't state one explicitly."
    )
    user_content = question
    if prior_context:
        # Ground truth from the actual preceding turn - see _extract_prior_tool_context.
        # Deliberately replaces the vaguer history-only instruction below: that one asked the
        # model to "carry over award_type from the history," which caused it to invent an
        # award_type that appeared in neither the history nor the new question (confirmed live).
        system += (
            " This is a follow-up in an ongoing conversation. The JSON below is the exact set of "
            "values already resolved by the preceding turn - reuse a field's value from it only "
            "when this new question doesn't say otherwise. Never set award_type, spending_level, "
            "or any other field unless it appears in this JSON or is explicitly stated in the new "
            "question - do not guess or default them.\n"
            f"Previously resolved values: {json.dumps(prior_context, default=str)}"
        )
        if history_block:
            user_content = f"{history_block}\n\nNew question: {question}"
    elif history_block:
        system += (
            " The user is continuing a prior download - carry over its agency from the "
            "history below unless this question names a different one; only the time period "
            "usually changes. Don't guess award_type or spending_level from a vague continuation - "
            "leave them unset unless the new question names one explicitly."
        )
        user_content = f"{history_block}\n\nNew question: {question}"
    try:
        response = _get_client().messages.create(
            model=MODEL,
            max_tokens=300,
            system=system,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": _EXTRACT_TOOL_NAME},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception:
        logger.exception("Download intent extraction call failed for question: %r", question)
        return None

    tool_use = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use is None:
        return None
    try:
        intent = DownloadIntent.model_validate(tool_use.input)
    except ValidationError:
        logger.warning("Download intent extraction returned unparseable input: %r", tool_use.input)
        return None
    if not intent.wants_download:
        return None
    has_year_range = intent.start_year is not None and intent.end_year is not None
    has_date_range = intent.start_date is not None and intent.end_date is not None
    if not has_year_range and not has_date_range:
        return None
    return intent


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
    for field in _CARRYOVER_FILTER_FIELDS:
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


def _build_single_award_citation(award_id: str, endpoint: str) -> ToolCitation:
    body = {"award_id": award_id, "file_format": "csv"}
    curl = f"curl -X POST '{BASE_URL}/api/v2/download/{endpoint}/' -H 'Content-Type: application/json' -d '{json.dumps(body)}'"
    return ToolCitation(
        tool_name=f"download_{endpoint}",
        parameters={"award_id": award_id},
        description=f"Single-award CSV download, {award_id}",
        curl=curl,
    )


def _handle_single_award_download(award_id: str, conversation_id: str):
    from .orchestrator import AgentResult

    endpoint = _award_id_endpoint(award_id)
    if endpoint is None:
        return None

    client = _get_usaspending_client()
    method = getattr(client, _SINGLE_AWARD_CLIENT_METHOD[endpoint])
    try:
        job: DownloadJobResponse = method(award_id)
        status = _poll_until_finished(client, job.file_name)
    except USASpendingAPIError as e:
        logger.warning("Single-award download pipeline failed for award_id %r: %s", award_id, e)
        return AgentResult(answer_text=f"This download failed: {e}.", conversation_id=conversation_id)

    citation = _build_single_award_citation(award_id, endpoint)
    download = DownloadSpec(
        file_name=status.file_name, url=status.file_url, status_url=job.status_url,
        status=status.status, total_rows=status.total_rows,
    )
    if status.status == "finished":
        answer_text = f"Your download is ready: {status.file_name}.\n{status.file_url}"
    elif status.status == "failed":
        answer_text = f"This download failed to generate: {status.message or 'no further detail from the API.'}"
    else:
        answer_text = (
            f"Your download is still generating ({status.file_name}). It'll appear at the link below "
            "once ready - check back shortly."
        )

    return AgentResult(
        answer_text=answer_text, conversation_id=conversation_id, downloads=[download], tool_citations=[citation]
    )


def handle_download_request(question: str, conversation_id: str, recent_messages: list | None = None):
    """Returns None to signal "fall through to the normal tool loop unchanged"."""
    from .orchestrator import AgentResult

    award_id = _resolve_single_award_id(question, recent_messages)
    if award_id is not None:
        single_award_result = _handle_single_award_download(award_id, conversation_id)
        if single_award_result is not None:
            return single_award_result
    elif _is_single_award_followup_phrase(question):
        # No tool in the loop below produces a file - say so rather than silently falling through.
        return AgentResult(
            answer_text=(
                "I can download that award once I know which one — look it up first (e.g. search "
                "for it or pull up its details), then ask me to download it."
            ),
            conversation_id=conversation_id,
        )

    unsupported = _unsupported_download_label(question)
    if unsupported is not None:
        return AgentResult(
            answer_text=(
                f"Downloading {unsupported} isn't supported yet — only award-level CSV "
                "exports are, for now."
            ),
            conversation_id=conversation_id,
        )

    # Only a real prior download gets the prose history block, whose prompt tells the extractor
    # the user is continuing one - prior_context stays ungated, being structured either way.
    history_block = _render_recent_exchanges(recent_messages) if _is_download_followup(recent_messages) else ""
    prior_context, dropped_filter_labels = (
        _extract_prior_tool_context(recent_messages) if recent_messages else (None, [])
    )
    intent = _extract_download_intent(question, history_block, prior_context)
    if intent is None:
        return None

    client = _get_usaspending_client()
    agency_name = None
    if intent.agency_raw:
        match = client.find_agency_by_name(intent.agency_raw)
        if match is None:
            return None
        agency_name = match.agency_name

    intent_context = _download_intent_context(intent, agency_name)
    carryover_filters = {field: getattr(intent, field) for field in _CARRYOVER_FILTER_FIELDS}

    try:
        # Placeholder when start_date is set - real scope is applied below by overwriting time_period.
        year_for_filters = int(intent.start_date[:4]) if intent.start_date else intent.start_year
        filters = _build_filters(
            client,
            agency_name,
            intent.time_period_type,
            year_for_filters,
            year_for_filters,
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
        logger.warning("Download pipeline failed for question %r: %s", question, e)
        return AgentResult(
            answer_text=f"This download failed: {e}.{_scope_caveat(dropped_filter_labels)}",
            conversation_id=conversation_id,
            download_intent_context=intent_context,
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
    answer_text += _scope_caveat(dropped_filter_labels, naics_note)

    return AgentResult(
        answer_text=answer_text, conversation_id=conversation_id, downloads=[download], tool_citations=[citation],
        download_intent_context=intent_context,
    )

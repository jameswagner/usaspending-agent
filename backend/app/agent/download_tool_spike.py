"""Unwired download tool shapes for the standalone spike."""
from __future__ import annotations

from typing import Annotated, Literal

from anthropic import beta_tool
from pydantic import Field

from backend.app.usaspending import TimePeriod, USASpendingAPIError

from .download_handler import (
    _DOWNLOAD_COLUMNS_BY_LEVEL,
    _SINGLE_AWARD_CLIENT_METHOD,
    _SPENDING_LEVEL_TO_API_ARRAY,
    _award_id_endpoint,
    _poll_until_finished,
)
from .response_shaping import DownloadSpec
from .singletons import _get_usaspending_client
from .tool_filters import _build_filters
from .tools._shared import _check_tool_call_budget, _record_tool_call

SpendingLevel = Literal["awards", "transactions", "subawards"]


def _complete_download(job, client, context: dict) -> str:
    status = _poll_until_finished(client, job.file_name)
    download = DownloadSpec(
        file_name=status.file_name,
        url=status.file_url,
        status_url=job.status_url,
        status=status.status,
        total_rows=status.total_rows,
    )
    _record_tool_call("download_tool", download, context)
    if status.status == "finished":
        return f"CSV download ready: {status.file_name}, {status.total_rows or 0} rows. {status.file_url}"
    if status.status == "failed":
        return f"This download failed to generate: {status.message or 'no further detail from the API.'}"
    return f"CSV download still generating: {status.file_name}. Status: {job.status_url}"


def _single_award(award_id: str) -> str:
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    endpoint = _award_id_endpoint(award_id)
    if endpoint is None:
        return "This download failed: award_id must begin CONT_AWD_, CONT_IDV_, or ASST_."
    client = _get_usaspending_client()
    try:
        job = getattr(client, _SINGLE_AWARD_CLIENT_METHOD[endpoint])(award_id)
        return _complete_download(job, client, {"award_id": award_id, "endpoint": endpoint})
    except USASpendingAPIError as exc:
        return f"This download failed: {exc}."


def _records(
    agency_name: str | None,
    start_fiscal_year: int | None,
    end_fiscal_year: int | None,
    start_date: str | None,
    end_date: str | None,
    award_type: Literal["contracts", "grants", "loans"] | None,
    spending_level: SpendingLevel,
    keywords: str | None,
    psc_code: str | None,
    naics_code: str | None,
) -> str:
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    dates = start_date is not None or end_date is not None
    years = start_fiscal_year is not None or end_fiscal_year is not None
    if dates == years or (dates and (start_date is None or end_date is None)) or (
        years and (start_fiscal_year is None or end_fiscal_year is None)
    ):
        return "This download failed: supply one complete fiscal-year range or one complete date range."
    client = _get_usaspending_client()
    if agency_name:
        match = client.find_agency_by_name(agency_name)
        if match is None:
            return f"This download failed: no agency found matching {agency_name}."
        agency_name = match.agency_name
    year = int(start_date[:4]) if dates else start_fiscal_year
    try:
        filters = _build_filters(
            client, agency_name, "fiscal", year, year if dates else end_fiscal_year,
            award_type=award_type, keywords=keywords, psc_code=psc_code,
            naics_code=naics_code, scope_required=False,
        )
        if dates:
            filters.time_period = [TimePeriod(start_date=start_date, end_date=end_date)]
        columns = _DOWNLOAD_COLUMNS_BY_LEVEL[spending_level]
        levels = _SPENDING_LEVEL_TO_API_ARRAY[spending_level]
        job = client.download_search(filters, columns, levels)
        return _complete_download(
            job, client,
            {"agency_name": agency_name, "start_fiscal_year": start_fiscal_year,
             "end_fiscal_year": end_fiscal_year, "start_date": start_date, "end_date": end_date,
             "award_type": award_type, "spending_level": spending_level,
             "keywords": keywords, "psc_code": psc_code, "naics_code": naics_code,
             "filters": filters, "columns": columns, "api_spending_level": levels},
        )
    except (USASpendingAPIError, ValueError) as exc:
        return f"This download failed: {exc}."


@beta_tool
def download_data(
    award_id: str | None = None,
    agency_name: str | None = None,
    start_fiscal_year: int | None = None,
    end_fiscal_year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    award_type: Literal["contracts", "grants", "loans"] | None = None,
    spending_level: SpendingLevel = "awards",
    keywords: Annotated[str | None, Field(description="Topic terms to match in the downloaded records; reuse a prior spending query's keywords.")] = None,
    psc_code: Annotated[str | None, Field(description="Four-character product or service code; reuse a prior spending query's validated PSC code.")] = None,
    naics_code: Annotated[str | None, Field(description="Two- to six-digit NAICS industry code; reuse a prior spending query's code.")] = None,
    file_format: Literal["csv"] = "csv",
) -> str:
    """Create a CSV download; use award_id alone for one award, or scope filters without award_id for records."""
    if award_id is not None:
        if any(value is not None for value in (
            agency_name, start_fiscal_year, end_fiscal_year, start_date, end_date,
            award_type, keywords, psc_code, naics_code,
        )) or spending_level != "awards":
            return "This download failed: award_id cannot be combined with record scope filters."
        return _single_award(award_id)
    return _records(
        agency_name, start_fiscal_year, end_fiscal_year, start_date, end_date,
        award_type, spending_level, keywords, psc_code, naics_code,
    )


@beta_tool
def download_single_award(award_id: str) -> str:
    """Create a CSV download of one known award using its full internal award ID."""
    return _single_award(award_id)


@beta_tool
def download_records(
    agency_name: str | None = None,
    start_fiscal_year: int | None = None,
    end_fiscal_year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    award_type: Literal["contracts", "grants", "loans"] | None = None,
    spending_level: SpendingLevel = "awards",
    keywords: Annotated[str | None, Field(description="Topic terms to match in the downloaded records; reuse a prior spending query's keywords.")] = None,
    psc_code: Annotated[str | None, Field(description="Four-character product or service code; reuse a prior spending query's validated PSC code.")] = None,
    naics_code: Annotated[str | None, Field(description="Two- to six-digit NAICS industry code; reuse a prior spending query's code.")] = None,
    file_format: Literal["csv"] = "csv",
) -> str:
    """Create a CSV of filtered award, transaction, or subaward records; use a fiscal-year or date range."""
    return _records(
        agency_name, start_fiscal_year, end_fiscal_year, start_date, end_date,
        award_type, spending_level, keywords, psc_code, naics_code,
    )

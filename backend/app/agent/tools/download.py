"""CSV download tools backed by the existing USAspending download pipeline."""
from __future__ import annotations

from typing import Annotated, Literal

from anthropic import beta_tool
from langsmith import traceable
from pydantic import Field

from backend.app.usaspending import TimePeriod, USASpendingAPIError

from ..download_handler import (
    _DOWNLOAD_COLUMNS_BY_LEVEL,
    _SINGLE_AWARD_CLIENT_METHOD,
    _SPENDING_LEVEL_TO_API_ARRAY,
    _award_id_endpoint,
    _poll_until_finished,
)
from ..response_shaping import DownloadSpec
from ..singletons import _get_usaspending_client
from ..tool_filters import (
    SpendingFilterParams,
    _build_filters,
    _pop_naics_disclosure,
    _validate_psc_code,
)
from ._shared import _check_tool_call_budget, _record_tool_call

SpendingLevel = Literal["awards", "transactions", "subawards"]


@traceable(run_type="tool", name="download_single_award_raw")
def download_single_award_raw(client, award_id: str, endpoint: str):
    """Submit a single-award download job."""
    return getattr(client, _SINGLE_AWARD_CLIENT_METHOD[endpoint])(award_id)


@traceable(run_type="tool", name="download_records_raw")
def download_records_raw(client, filters, columns: list[str], levels: list[str]):
    """Submit a filtered-record download job."""
    return client.download_search(filters, columns, levels)


def _complete_download(job, client, tool_name: str, context: dict) -> str:
    status = _poll_until_finished(client, job.file_name)
    download = DownloadSpec(
        file_name=status.file_name,
        url=status.file_url,
        status_url=job.status_url,
        status=status.status,
        total_rows=status.total_rows,
    )
    _record_tool_call(tool_name, download, context)
    if status.status == "finished":
        return f"CSV download ready: {status.file_name}, {status.total_rows or 0} rows. {status.file_url}"
    if status.status == "failed":
        return f"This download failed to generate: {status.message or 'no further detail from the API.'}"
    return f"CSV download still generating: {status.file_name}. Status: {job.status_url}"


@beta_tool
def download_single_award(award_id: str) -> str:
    """Create a CSV of one known award using its full CONT_AWD_, CONT_IDV_, or ASST_ internal ID."""
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    endpoint = _award_id_endpoint(award_id)
    if endpoint is None:
        return "This download failed: award_id must begin CONT_AWD_, CONT_IDV_, or ASST_."
    client = _get_usaspending_client()
    try:
        job = download_single_award_raw(client, award_id, endpoint)
        return _complete_download(job, client, "download_single_award", {"award_id": award_id, "endpoint": endpoint})
    except USASpendingAPIError as exc:
        return f"This download failed: {exc}."


def _records(args: dict) -> str:
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    if args["recipient_id"] is not None:
        return "This download failed: the download endpoint cannot filter by recipient ID; the file would include a wider scope."
    if args["psc_codes"] is not None and (not args["psc_codes"] or args["psc_code"] is not None):
        return "This download failed: supply either psc_code or a nonempty psc_codes list."
    dates = args["start_date"] is not None or args["end_date"] is not None
    years = args["start_fiscal_year"] is not None or args["end_fiscal_year"] is not None
    if dates == years or (dates and (args["start_date"] is None or args["end_date"] is None)) or (
        years and (args["start_fiscal_year"] is None or args["end_fiscal_year"] is None)
    ):
        return "This download failed: supply one complete fiscal-year range or one complete date range."
    client = _get_usaspending_client()
    agency_name = args["agency_name"]
    try:
        if agency_name:
            match = client.find_agency_by_name(agency_name)
            if match is None:
                return f"This download failed: no agency found matching {agency_name}."
            agency_name = match.agency_name
        year = int(args["start_date"][:4]) if dates else args["start_fiscal_year"]
        filter_args = {
            key: args[key] for key in SpendingFilterParams.__annotations__
            if key != "recipient_id" and args.get(key) is not None
        }
        filters = _build_filters(
            client, agency_name, "fiscal", year,
            year if dates else args["end_fiscal_year"], scope_required=False, **filter_args,
        )
        naics_note = _pop_naics_disclosure()
        if args["psc_codes"] is not None:
            filters.psc_codes = list(dict.fromkeys(_validate_psc_code(code) for code in args["psc_codes"]))
        if dates:
            filters.time_period = [TimePeriod(start_date=args["start_date"], end_date=args["end_date"])]
        level = args["spending_level"]
        columns = _DOWNLOAD_COLUMNS_BY_LEVEL[level]
        api_levels = (
            [level] if not args["include_subawards"] and level != "subawards"
            else _SPENDING_LEVEL_TO_API_ARRAY[level]
        )
        job = download_records_raw(client, filters, columns, api_levels)
        context = {
            **{key: value for key, value in args.items() if value is not None},
            "agency_name": agency_name,
            "filters": filters,
            "columns": columns,
            "api_spending_level": api_levels,
        }
        result = _complete_download(job, client, "download_records", context)
        return f"{result}\n\nNote: {naics_note}" if naics_note else result
    except (USASpendingAPIError, ValueError) as exc:
        return f"This download failed: {exc}."


@beta_tool
def download_records(
    agency_name: str | None = None,
    start_fiscal_year: Annotated[int | None, Field(description="First fiscal year, inclusive; pair with end_fiscal_year.")] = None,
    end_fiscal_year: Annotated[int | None, Field(description="Last fiscal year, inclusive; pair with start_fiscal_year.")] = None,
    start_date: Annotated[str | None, Field(description="YYYY-MM-DD first calendar date, inclusive; January 2024 begins 2024-01-01.")] = None,
    end_date: Annotated[str | None, Field(description="YYYY-MM-DD last calendar date, inclusive; January 2024 ends 2024-01-31.")] = None,
    award_type: Annotated[str | None, Field(description="Award type only when explicitly requested; omit for all types.")] = None,
    spending_level: Annotated[SpendingLevel, Field(description="Record level: awards by default, transactions or subawards only when requested.")] = "awards",
    include_subawards: Annotated[bool, Field(description="Defaults to false for one awards-only or transactions-only CSV; set true only when the user requests bundled subawards.")] = False,
    recipient_name: str | None = None,
    recipient_id: Annotated[str | None, Field(description="Unsupported by the download endpoint; passing it returns an error to prevent a broader file.")] = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    performed_in_state: str | None = None,
    recipient_in_state: str | None = None,
    performed_in_county: str | None = None,
    recipient_in_county: str | None = None,
    performed_in_city: str | None = None,
    recipient_in_city: str | None = None,
    performed_in_zip: str | None = None,
    recipient_in_zip: str | None = None,
    performed_in_district: str | None = None,
    recipient_in_district: str | None = None,
    keywords: Annotated[str | None, Field(description="Topic terms; reuse a prior spending query's keywords.")] = None,
    date_type: str | None = None,
    place_of_performance_scope: str | None = None,
    recipient_scope: str | None = None,
    naics_code: str | None = None,
    psc_code: Annotated[str | None, Field(description="One four-character PSC code; use psc_codes for several codes in one download.")] = None,
    psc_codes: Annotated[list[str] | None, Field(description="Several four-character PSC codes in one download job; matches any listed code, so do not make one call per code.")] = None,
    cfda_program: str | None = None,
    award_id: Annotated[str | None, Field(description="PIID, FAIN, or URI record filter; use download_single_award for a full internal ID.")] = None,
    recipient_type: str | None = None,
    description: str | None = None,
    tas_code: str | None = None,
    federal_account: str | None = None,
    def_codes: list[str] | None = None,
    contract_pricing_type: list[str] | None = None,
    set_aside_type: list[str] | None = None,
    extent_competed_type: list[str] | None = None,
    file_format: Literal["csv"] = "csv",
) -> str:
    """Create one filtered CSV download job; psc_codes combines codes in one file, and awards-only is the default."""
    return _records(locals())

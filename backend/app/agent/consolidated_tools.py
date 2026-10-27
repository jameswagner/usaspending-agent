"""An alternative, consolidated tool surface for the same agent: fewer tools,
each with a section/kind/group_by parameter, for comparison against
LANGGRAPH_TOOLS via AGENT_TOOLSET=consolidated or the compare_raw_api harness.

Every tool here delegates to the existing @beta_tool functions' .func, so
call recording, citations, charts, and follow-ups (all keyed on the
underlying tool's own recorded name) work unchanged. Kept tools whose
docstrings mention a tool that only exists in the current surface get those
references rewritten (_rename_refs), in their schema and in their results.
"""
from __future__ import annotations

import functools
import inspect
import re
import types
from typing import Literal

from langchain_core.tools import tool as _lc_tool
from pydantic import ConfigDict, Field

from .langgraph_tools import _ARITHMETIC_TOOLS, _reject_invalid_args, _with_budget_guard
from .tool_filters import AWARD_TYPE_GROUPS, _normalize_award_type
from .tools import (
    download_records,
    download_single_award,
    get_agency_award_breakdown,
    get_agency_budget,
    get_award_details,
    get_award_funding_breakdown,
    get_award_subawards,
    get_award_transaction_history,
    get_award_type_breakdown,
    get_disaster_spending_overview,
    get_recipient_children,
    get_recipient_details,
    get_spending_explorer_breakdown,
    list_top_agencies_by_budget,
    lookup_agency,
    query_spending,
    resolve_budget_function,
    resolve_cfda_program,
    resolve_county_fips,
    resolve_naics_code,
    resolve_psc_code,
    search_guide,
    search_recipients,
)
from .tools.spending.consolidated import SpendingGroupBy

_RENAMES = {
    "get_agency_budget_by_subcomponent": 'agency_profile(section="budget_by_bureau")',
    "get_award_transaction_history": 'award_detail(section="transactions")',
    "get_award_funding_breakdown": 'award_detail(section="funding")',
    "get_agency_award_breakdown": 'agency_profile(section="awards_by_sub_agency")',
    "list_top_agencies_by_budget": 'agency_profile(section="rank_all_agencies")',
    "get_spending_by_geography": 'query_spending(group_by="geography")',
    "get_award_type_breakdown": 'query_spending(group_by="award_type")',
    "get_spending_by_category": "query_spending(group_by=<category>)",
    "resolve_budget_function": 'resolve_entity(kind="budget_function")',
    "get_spending_over_time": 'query_spending(group_by="time")',
    "get_recipient_children": 'recipient_profile(section="children")',
    "get_recipient_details": 'recipient_profile(section="details")',
    "resolve_cfda_program": 'resolve_entity(kind="cfda")',
    "search_transactions": 'query_spending(level="transaction")',
    "get_award_subawards": 'award_detail(section="subawards")',
    "resolve_county_fips": 'resolve_entity(kind="county")',
    "resolve_naics_code": 'resolve_entity(kind="naics")',
    "get_award_details": 'award_detail(section="summary")',
    "get_agency_budget": 'agency_profile(section="budget")',
    "search_recipients": 'resolve_entity(kind="recipient")',
    "search_subawards": 'query_spending(level="subaward")',
    "resolve_psc_code": 'resolve_entity(kind="psc")',
    "search_awards": 'query_spending(level="award")',
    "lookup_agency": 'agency_profile(section="overview")',
}
_RENAME_PATTERN = re.compile(r"\b(" + "|".join(sorted(_RENAMES, key=len, reverse=True)) + r")\b")


def _rename_refs(text: str) -> str:
    return _RENAME_PATTERN.sub(lambda m: _RENAMES[m.group(1)], text)


def _renaming_results(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        return _rename_refs(result) if isinstance(result, str) else result

    return wrapped


def _with_renamed_doc(func):
    copy = types.FunctionType(func.__code__, func.__globals__, func.__name__, func.__defaults__, func.__closure__)
    copy.__kwdefaults__ = func.__kwdefaults__
    functools.update_wrapper(copy, func)
    copy.__doc__ = _rename_refs(func.__doc__ or "")
    return copy


def _bind(func, parse_docstring: bool = True):
    return _lc_tool(_with_budget_guard(_renaming_results(func)), parse_docstring=parse_docstring)


def _only_for(section: str, allowed: set[str], **given) -> str | None:
    stray = sorted(name for name, value in given.items() if value not in (None, False) and name not in allowed)
    if stray:
        return f"This query failed: {', '.join(stray)} does not apply to section={section!r}."
    return None


ResolveKind = Literal["recipient", "naics", "psc", "cfda", "county", "budget_function"]


def resolve_entity(kind: ResolveKind, query: str) -> str:
    """Turn a plain-English name into the exact code or ID a filter needs. Call this before scoping a spending question by a recipient, industry, product/service, assistance program, county, or budget function the user described in words rather than as a code. Agencies need no resolving: pass the agency's name directly as agency_name; for what an agency is or its toptier code, use agency_profile(section="overview").

    NAICS, PSC, CFDA, and budget-function results are semantic matches against official titles and descriptions, not authoritative lookups. Confirm the returned title actually matches what the user meant, and when presenting a match say it is the closest matching code, not a confirmed exact one. query_spending's naics_code already auto-resolves a single confident match, so resolve NAICS here only when the description is ambiguous.

    A recipient name is often genuinely ambiguous: one company name can map to several recipient_ids (parent, subsidiaries, historical registrations). Every candidate is shown; for a "how much has this company received in total" question prefer the "parent"-level candidate, which rolls up all of its children. Ask the user if it is still unclear. An exact UEI or DUNS returns a single precise match.

    Args:
        kind: What to resolve. "recipient": a company/organization/individual's recipient_id, by name, UEI, or DUNS, for query_spending's recipient_id or recipient_profile. "naics": NAICS industry code(s) for an industry/business description, for naics_code; query="sectors" lists the official top-level sectors, and a query starting with a 2- or 4-digit code lists the official industries under it. "psc": Product and Service Code(s) for a description of what is being bought, for psc_code. "cfda": CFDA/Assistance Listing number(s) for a grant/loan/assistance program description, for cfda_program. "county": a county's 3-digit FIPS code, for performed_in_county/recipient_in_county - always pair it with its state, since county names and codes repeat across states. "budget_function": budget function/subfunction code(s) for a budget category such as "National Defense" or "Medicare", for get_spending_explorer_breakdown's budget_function/budget_subfunction filters.
        query: The name or description to resolve, e.g. "Boeing", "custom software development", "office furniture", "rural broadband grants", "Yavapai County", or "Health".
    """
    if kind == "recipient":
        return search_recipients.func(keyword=query)
    resolver = {
        "naics": resolve_naics_code,
        "psc": resolve_psc_code,
        "cfda": resolve_cfda_program,
        "county": resolve_county_fips,
        "budget_function": resolve_budget_function,
    }[kind]
    return resolver.func(description=query)


AgencySection = Literal[
    "overview", "budget", "budget_by_bureau", "awards_by_sub_agency",
    "awards_by_category", "award_activity", "rank_all_agencies",
]
AgencyAwardType = Literal[
    "contracts", "grants", "loans", "bpa_call", "purchase_order", "delivery_order", "definitive_contract",
    "idv", "idv_gwac", "idv_idc", "idv_fss", "idv_boa", "idv_bpa",
    "direct_loan", "guaranteed_loan", "block_grant", "formula_grant", "project_grant",
    "cooperative_agreement", "insurance", "other_financial_assistance", "direct_payment_specified",
    "direct_payment_unrestricted",
]


def agency_profile(
    section: AgencySection,
    agency_name: str | None = None,
    fiscal_year: int | None = None,
    end_fiscal_year: int | None = None,
    include_period_breakdown: bool = False,
    bureau: str | None = None,
    sub_agency: str | None = None,
    award_type: AgencyAwardType | None = None,
    include_offices: bool = False,
    limit: int = 10,
) -> str:
    """Everything about one federal agency as an organization: what it is, its appropriated budget, its bureaus, and its award obligations and counts by sub-agency or award category - or a ranking of all agencies by budget.

    Two different kinds of number live here; never present one as the other. The budget sections ("budget", "budget_by_bureau", "rank_all_agencies") report budgetary resources, obligations, and outlays from the agency's accounts: use them for "what is X's budget" or "how much money does X have", never query_spending. The award sections ("awards_by_sub_agency", "awards_by_category", "award_activity") report obligations on specific contracts, grants, and loans, with transaction and new-award counts by sub-agency. For a count of awards by type under any other filter (a place, a recipient, new awards only), use query_spending(group_by="award_type"). An agency's budgetary resources and its award obligations for the same year are different figures.

    Args:
        section: "overview": what the agency is - toptier code, abbreviation, mission, website, budget justification link, subtier count. "budget": yearly budgetary resources, obligations, and outlays for fiscal_year through end_fiscal_year (data from FY2017 on). "budget_by_bureau": the same figures for one fiscal year broken down by sub-component/bureau (e.g. NIH or CDC within HHS), or one bureau's own federal accounts when bureau is set. "awards_by_sub_agency": one fiscal year's award obligations plus transaction and new-award counts by sub-agency. "awards_by_category": one fiscal year's award obligations split by contracts/IDVs/grants/loans/direct payments/other. "award_activity": the agency's whole-agency award obligations and transaction count for one fiscal year (no new-award count; a top-tier agency only, not a sub-agency like NIH). "rank_all_agencies": all agencies ranked by budgetary resources with their share of the federal total, current fiscal year only (agency_name not used).
        agency_name: The agency's name, e.g. "National Science Foundation" or "NSF". Required for every section except "rank_all_agencies". A sub-agency name also works for the award sections.
        fiscal_year: The fiscal year, e.g. 2024 for FY2024. Required for every section except "overview" and "rank_all_agencies". For "budget" it is the first year of the range.
        end_fiscal_year: "budget" only. Last fiscal year of the range; defaults to fiscal_year.
        include_period_breakdown: "budget" only. True only when the question is about when during the year money was obligated; adds per-period figures.
        bureau: "budget_by_bureau" only. One bureau's name, matched against the bureau names this section returns without it, to list that bureau's federal accounts.
        sub_agency: "awards_by_sub_agency" only. One sub-agency (e.g. "Food and Nutrition Service" under USDA), to rank its awarding offices instead of listing every sub-agency.
        award_type: "awards_by_sub_agency" or "award_activity" only. Restrict to one award type or bucket; omit for all award types.
        include_offices: "awards_by_sub_agency" only. True to also list each sub-agency's awarding offices; only when the question asks about offices.
        limit: "rank_all_agencies" only. Max agencies to return (default 10).
    """
    given = {
        "agency_name": agency_name, "fiscal_year": fiscal_year, "end_fiscal_year": end_fiscal_year,
        "include_period_breakdown": include_period_breakdown, "bureau": bureau, "sub_agency": sub_agency,
        "award_type": award_type, "include_offices": include_offices,
    }
    allowed = {
        "overview": {"agency_name"},
        "budget": {"agency_name", "fiscal_year", "end_fiscal_year", "include_period_breakdown"},
        "budget_by_bureau": {"agency_name", "fiscal_year", "bureau"},
        "awards_by_sub_agency": {"agency_name", "fiscal_year", "sub_agency", "award_type", "include_offices"},
        "awards_by_category": {"agency_name", "fiscal_year"},
        "award_activity": {"agency_name", "fiscal_year", "award_type"},
        "rank_all_agencies": set(),
    }[section]
    if (error := _only_for(section, allowed, **given)) is not None:
        return error
    if section == "rank_all_agencies":
        return list_top_agencies_by_budget.func(limit=limit)
    if agency_name is None:
        return f"This query failed: agency_name is required for section={section!r}."
    if section == "overview":
        return lookup_agency.func(name=agency_name)
    if fiscal_year is None:
        return f"This query failed: fiscal_year is required for section={section!r}."
    if section == "budget":
        return get_agency_budget.func(
            agency_name=agency_name, start_fiscal_year=fiscal_year, end_fiscal_year=end_fiscal_year or fiscal_year,
            include_period_breakdown=include_period_breakdown,
        )
    if section == "budget_by_bureau":
        return get_agency_budget.func(
            agency_name=agency_name, start_fiscal_year=fiscal_year, end_fiscal_year=fiscal_year,
            group_by="sub_component", bureau=bureau,
        )
    group_by = {section_name: mode for mode, section_name in _AWARD_SECTION_BY_GROUP_BY.items()}[section]
    result = get_agency_award_breakdown.func(
        agency_name=agency_name, fiscal_year=fiscal_year, award_type=award_type,
        include_offices=include_offices, group_by=group_by, sub_agency=sub_agency,
    )
    for mode, section_name in _AWARD_SECTION_BY_GROUP_BY.items():
        result = result.replace(f'group_by="{mode}"', f'section="{section_name}"')
    return result


_AWARD_SECTION_BY_GROUP_BY = {
    "sub_agency": "awards_by_sub_agency", "award_category": "awards_by_category", "whole_agency": "award_activity",
}

AwardSection = Literal["summary", "transactions", "subawards", "funding"]


def award_detail(
    award_id: str,
    section: AwardSection = "summary",
    include_child_orders: bool = False,
    limit: int = 10,
) -> str:
    """Details about ONE specific award (contract, IDV, grant, loan, or other assistance) already found via query_spending - never for browsing or listing awards, which query_spending does.

    Args:
        award_id: The internal_id shown alongside a query_spending record (e.g. "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-"), or from a prior award_detail parent-IDV line - NOT the plain PIID/FAIN shown next to it. Never guess or construct one.
        section: "summary" (default): description, dates, competition data, recipient, funding totals, and parent IDV. "transactions": the award's individual modifications/transactions, most recent first - an empty result can also mean a wrong award_id, so don't treat it alone as proof of no transactions. "subawards": the subawards issued under this prime award (for subawards across many awards, use query_spending(level="subaward") instead). "funding": which Treasury Account Symbols, object classes, program activities, and DEFCs funded the award - a separate, later-timed account-level source from the summary's own total, not guaranteed to reconcile; get the summary first for the headline totals.
        include_child_orders: "summary" only. True only for an IDV (contract vehicle) when the question is how much has been ordered under it; costs a second API call.
        limit: "transactions", "subawards", or "funding" only. Max rows to return (default 10).
    """
    if section != "summary" and include_child_orders:
        return f"This query failed: include_child_orders does not apply to section={section!r}."
    if section == "summary":
        return get_award_details.func(award_id=award_id, include_child_orders=include_child_orders)
    handler = {
        "transactions": get_award_transaction_history,
        "subawards": get_award_subawards,
        "funding": get_award_funding_breakdown,
    }[section]
    return handler.func(award_id=award_id, limit=limit)


RecipientSection = Literal["details", "children"]


def recipient_profile(
    recipient_id: str,
    section: RecipientSection = "details",
    year: str = "all",
    limit: int = 25,
) -> str:
    """Profile of ONE already-resolved recipient - get its recipient_id from resolve_entity(kind="recipient") first; never guess one.

    Args:
        recipient_id: The exact recipient_id from resolve_entity(kind="recipient"), e.g. "419ccd27-d6f4-d363-aeaf-b9e2c3ae6f5d-P".
        section: "details" (default): identity (name, alternate names, UEI/DUNS), parent relationship, address, business types, and total federal transaction amount for the period. "children": the child recipients (subsidiaries/registrations) rolling up into a "parent"-level recipient's total, largest first - a "standalone" or "child"-level recipient has none.
        year: A fiscal year (e.g. "2023"), "all" (default - the recipient's entire history), or "latest" (trailing 12 months, the same window resolve_entity's amounts already show).
        limit: "children" only. Max children to return (default 25).
    """
    if section == "details":
        return get_recipient_details.func(recipient_id=recipient_id, year=year)
    return get_recipient_children.func(recipient_id=recipient_id, year=year, limit=limit)


ConsolidatedGroupBy = Literal[(*SpendingGroupBy.__args__, "award_type")]

_AWARD_TYPE_COUNT_IGNORED = {"level", "group_by", "limit", "sort_by", "time_grouping"}
_AWARD_TYPE_COUNT_PARAMS = set(inspect.signature(get_award_type_breakdown.func).parameters)

_AWARD_TYPE_COUNT_DOC = (
    ' "award_type" for the COUNT of awards by type (Contracts, Contract IDVs, Grants, Direct Payments, '
    "Loans, Other) - use it for \"how many contracts vs. grants\" or award-type-mix questions instead of "
    "querying once per award type; it needs no scoping filter (a bare year range is a government-wide split), "
    "ignores level and award_type (the counts are already split by type), and cannot be combined with "
    "scope/geo_layer, tas_code, federal_account, def_codes, or the contract-only filters."
)

_AWARD_TYPE_SCOPE_DOC = (
    " On a breakdown (group_by set), set award_type only when the question is about one award type: "
    "it narrows every figure to that type."
)
_CONTRACT_ONLY_GROUP_BYS = {"naics", "psc"}


def _is_assistance(award_type: str) -> bool:
    codes = AWARD_TYPE_GROUPS.get(_normalize_award_type(award_type), [])
    contract_codes = set(AWARD_TYPE_GROUPS["contracts"]) | set(AWARD_TYPE_GROUPS["idv"])
    return bool(codes) and not contract_codes & set(codes)


def _query_spending(**kwargs) -> str:
    group_by, award_type = kwargs.get("group_by"), kwargs.get("award_type")
    if group_by in _CONTRACT_ONLY_GROUP_BYS and award_type and _is_assistance(award_type):
        return (
            f"This query failed: {group_by.upper()} codes exist only on contracts and IDVs, so a {group_by} "
            f"breakdown of award_type={award_type!r} is always empty. Use group_by='cfda' for assistance "
            "programs, or omit award_type to cover every award type."
        )
    if group_by != "award_type":
        return query_spending.func(**kwargs)
    note = ""
    if award_type is not None:
        kwargs = {**kwargs, "award_type": None}
        note = (
            f"Note: award_type={award_type!r} was not applied as a filter. group_by='award_type' already "
            "splits the counts by type, so read the matching line below.\n\n"
        )
    unsupported = sorted(
        name for name, value in kwargs.items()
        if value is not None and name not in _AWARD_TYPE_COUNT_PARAMS | _AWARD_TYPE_COUNT_IGNORED
    )
    if unsupported:
        return f"This query failed: {', '.join(unsupported)} cannot be combined with group_by='award_type'."
    return note + get_award_type_breakdown.func(**{k: v for k, v in kwargs.items() if k in _AWARD_TYPE_COUNT_PARAMS})


def _build_query_spending_tool():
    tool = _bind(_with_renamed_doc(query_spending.func))
    base_schema = tool.args_schema
    base_doc = base_schema.model_fields["group_by"].description or ""
    award_type_field = base_schema.model_fields["award_type"]

    class ConsolidatedQuerySpendingArgs(base_schema):
        model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
        group_by: ConsolidatedGroupBy | None = Field(None, description=base_doc + _AWARD_TYPE_COUNT_DOC)
        award_type: award_type_field.annotation = Field(
            None, description=(award_type_field.description or "") + _AWARD_TYPE_SCOPE_DOC
        )

    tool.args_schema = ConsolidatedQuerySpendingArgs
    tool.func = _with_budget_guard(_renaming_results(_query_spending))
    tool.handle_validation_error = lambda error: _reject_invalid_args(error, "This query failed:")
    return tool


def _build_download_tool(beta_tool):
    tool = _bind(_with_renamed_doc(beta_tool.func), parse_docstring=False)
    tool.args_schema = type(
        f"Strict{tool.name}Args",
        (tool.args_schema,),
        {"model_config": ConfigDict(arbitrary_types_allowed=True, extra="forbid")},
    )
    tool.handle_validation_error = lambda error: _reject_invalid_args(error, "This download failed:")
    return tool


# Which current-surface tools each consolidated tool reaches, for the coverage test.
COVERS: dict[str, set[str]] = {
    "resolve_entity": {
        "search_recipients", "resolve_naics_code", "resolve_psc_code", "resolve_cfda_program",
        "resolve_county_fips", "resolve_budget_function",
    },
    "agency_profile": {"lookup_agency", "get_agency_budget", "get_agency_award_breakdown", "list_top_agencies_by_budget"},
    "award_detail": {"get_award_details", "get_award_transaction_history", "get_award_subawards", "get_award_funding_breakdown"},
    "recipient_profile": {"get_recipient_details", "get_recipient_children"},
    "query_spending": {"query_spending", "get_award_type_breakdown"},
    "get_spending_explorer_breakdown": {"get_spending_explorer_breakdown"},
    "get_disaster_spending_overview": {"get_disaster_spending_overview"},
    "download_records": {"download_records"},
    "download_single_award": {"download_single_award"},
    "search_guide": {"search_guide"},
}

CONSOLIDATED_TOOLS = [
    _bind(search_guide.func),
    _bind(resolve_entity),
    _build_query_spending_tool(),
    _bind(agency_profile),
    _bind(_with_renamed_doc(get_spending_explorer_breakdown.func)),
    _bind(award_detail),
    _bind(recipient_profile),
    _bind(_with_renamed_doc(get_disaster_spending_overview.func)),
    _build_download_tool(download_records),
    _build_download_tool(download_single_award),
] + [_lc_tool(at.func, parse_docstring=True) for at in _ARITHMETIC_TOOLS]

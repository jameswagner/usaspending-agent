"""get_award_details and its IDV child-order rollup - full detail on one
specific award (contract, IDV, grant, loan, or other financial
assistance), as opposed to spending.py's aggregate/list tools. Also
get_award_subawards, listing one award's own subawards. Also
get_award_funding_breakdown, the Federal Account/TAS funding breakdown for
one specific award. Also get_award_transaction_history, the
modification-by-modification transaction history for one specific award.
"""
from __future__ import annotations

import logging
from typing import Any

from anthropic import beta_tool
from langsmith import traceable

from backend.app.usaspending import (
    AwardFundingResponse,
    IDVAmountsResponse,
    TransactionHistoryResponse,
    USASpendingAPIError,
)

from ..singletons import _get_usaspending_client
from ._shared import (
    _check_tool_call_budget,
    _record_tool_call,
    _truncation_note,
    _wrap_untrusted,
)

logger = logging.getLogger(__name__)

# generated_unique_award_id's own real prefixes (live-verified 2026-09-15,
# #146): CONT_AWD_/CONT_IDV_ for contracts/IDVs, ASST_NON_ for financial
# assistance (grants/loans/direct payments/other). A value that starts
# with neither is almost certainly a plain PIID/FAIN passed by mistake -
# used only to decide whether to append the IDV-resolution hint below, not
# to validate/reject the input (the live API is still the source of truth
# for whether it resolves).
_INTERNAL_ID_PREFIXES = ("CONT_", "ASST_")


def _unresolved_award_id_hint(award_id: str) -> str:
    """Appended to a get_award_details failure message when award_id
    doesn't look like a real internal_id - most commonly a plain PIID/FAIN
    passed by mistake, including the #146 case (a plain IDV PIID). Empty
    string when award_id already has a real prefix, since the failure is
    then something else (a bad/nonexistent internal_id) this hint doesn't
    apply to."""
    if award_id.upper().startswith(_INTERNAL_ID_PREFIXES):
        return ""
    return (
        " This tool requires the hash-style internal_id (e.g. "
        "'CONT_AWD_...' for a contract, 'CONT_IDV_...' for a contract vehicle/IDV, "
        "'ASST_NON_...' for a grant/loan/other assistance award), not a plain "
        "Award ID/PIID/FAIN. Resolve it first via search_awards — if this is a "
        "contract vehicle (IDV) rather than a plain contract, call "
        "search_awards(award_type=\"idv\", award_id=<PIID>) — then use the "
        "internal_id shown in that result."
    )


@traceable(run_type="tool", name="get_award_details_raw")
def get_award_details_raw(award_id: str) -> dict[str, Any]:
    """Call the API once, return the raw response dict. Raises
    USASpendingAPIError if award_id doesn't resolve - most commonly
    because a plain PIID/FAIN was passed instead of the hash-style
    internal_id search_awards's own output now carries alongside it (see
    client.get_award's docstring)."""
    client = _get_usaspending_client()
    return client.get_award(award_id)


@traceable(run_type="tool", name="get_idv_amounts_raw")
def get_idv_amounts_raw(award_id: str) -> IDVAmountsResponse:
    """Call the API once, return the structured child/grandchild-order
    rollup. Raises USASpendingAPIError on failure - see
    IDVAmountsResponse's docstring for what this covers and why it's a
    separate call from get_award_details_raw."""
    client = _get_usaspending_client()
    return client.get_idv_amounts(award_id)


@traceable(run_type="tool", name="get_award_funding_breakdown_raw")
def get_award_funding_breakdown_raw(award_id: str, limit: int = 10) -> AwardFundingResponse:
    """Call the API once, return the structured Federal Account Funding
    rows. Raises USASpendingAPIError on failure - see
    USASpendingClient.get_award_funding's docstring for why this is a
    separate call from get_award_details_raw rather than merged into it."""
    client = _get_usaspending_client()
    return client.get_award_funding(award_id, limit=limit)


@traceable(run_type="tool", name="get_award_transaction_history_raw")
def get_award_transaction_history_raw(award_id: str, limit: int = 10) -> TransactionHistoryResponse:
    """Call the API once, return the structured transaction/modification
    rows. Raises USASpendingAPIError on failure - see
    USASpendingClient.get_award_transaction_history's docstring for the
    plain-PIID-returns-empty-not-an-error gotcha."""
    client = _get_usaspending_client()
    return client.get_award_transaction_history(award_id, limit=limit)


def _agency_label(agency: dict[str, Any] | None) -> str:
    """awarding_agency/funding_agency are Agency objects (id,
    has_agency_page, toptier_agency, subtier_agency, office_agency_name) -
    this pulls just the toptier name/abbreviation, the same trim applied
    to every other nested object here (recipient/location full addresses,
    executive compensation, etc. are all left out entirely - see the
    award-detail design walkthrough)."""
    if not agency:
        return "N/A"
    toptier = agency.get("toptier_agency") or {}
    name = toptier.get("name")
    abbreviation = toptier.get("abbreviation")
    if name and abbreviation:
        return f"{name} ({abbreviation})"
    return name or abbreviation or "N/A"


def _location_label(location: dict[str, Any] | None, *, full: bool = True) -> str:
    """full=False shows state only - used for record_type 1/3 recipients
    (aggregate/PII-redacted), where the live API still returns city/county/
    zip-level detail even though the recipient's own identity is withheld.
    Showing that full granularity here would undercut the redaction's own
    purpose, so this app deliberately narrows it to state-level for those
    two cases rather than forwarding whatever the API happens to return."""
    if not location:
        return "N/A"
    state = location.get("state_name") or location.get("state_code")
    if not full:
        return state or "N/A"
    city = location.get("city_name")
    if city and state:
        return f"{city}, {state}"
    return state or location.get("country_name") or "N/A"


def _congressional_district_label(location: dict[str, Any] | None) -> str | None:
    """"CD-##" from state_code + congressional_code - None when either half is missing, since a bare "-06" or "MD-" is worse than omitting the line."""
    if not location:
        return None
    state = location.get("state_code")
    district = location.get("congressional_code")
    if not state or not district:
        return None
    return f"{state}-{district}"


def _format_period_of_performance(pop: dict[str, Any] | None) -> str | None:
    """None (not a placeholder string) when neither date is present - a
    bare "? to ?" fallback would just be noise, so callers skip the line
    entirely rather than printing a placeholder."""
    if not pop or (not pop.get("start_date") and not pop.get("end_date")):
        return None
    start = pop.get("start_date") or "?"
    end = pop.get("end_date") or "?"
    label = f"{start} to {end}"
    potential = pop.get("potential_end_date")
    if potential and potential != end:
        label += f" (potential end date {potential})"
    return label


def _format_contract_or_idv(
    data: dict[str, Any], child_order_rollup: IDVAmountsResponse | None = None
) -> str:
    """Shared formatting for ContractResponse and IDVResponse - identical
    field shape per award_id.md (IDVResponse just adds a nullable
    total_outlay this app doesn't surface). Deliberately trims the ~35
    top-level fields (plus ~60 on latest_transaction_contract_data) down
    to money, dates, parties, parent-award linkage, and the handful of
    procurement-detail fields an analyst actually asks about - not the FAR
    policy-code fields, not executive compensation, and not the
    Treasury-account-level totals (total_account_obligation, the *_by_defc
    arrays), which come from a different, separately-timed DATA Act
    submission (File C) than this award's own total_obligation (File D2)
    and would confuse if mixed into one answer unlabeled. total_outlay is
    shown despite also being File-C-sourced - it's award-scoped, not the
    DEFC-bucketed total_account_outlay excluded above.

    child_order_rollup is only ever passed for category == "idv"
    (get_award_details only fetches it when include_child_orders is set
    and the award is an IDV) - it replaces the generic "$0 doesn't mean
    nothing happened" caveat with the real child/grandchild-order totals
    from GET /api/v2/idvs/amounts/{award_id}/."""
    lines = [f"{data.get('type_description', 'Unknown type')} ({data.get('piid', 'unknown PIID')})"]
    if data.get("description"):
        lines.append(f"Description: {data['description']}")
    total_obligation = data.get("total_obligation") or 0
    ceiling = data.get("base_and_all_options") or 0
    lines.append(f"Total obligated: ${total_obligation:,.2f}, ceiling (base + all options): ${ceiling:,.2f}")
    current_award_amount = data.get("base_exercised_options")
    if current_award_amount is not None:
        lines.append(f"Current award amount (base + exercised options): ${current_award_amount:,.2f}")
    outlay = data.get("total_outlay")
    if outlay is not None:
        lines.append(f"Outlayed: ${outlay:,.2f}")
    if data.get("date_signed"):
        lines.append(f"Date signed: {data['date_signed']}")
    pop_label = _format_period_of_performance(data.get('period_of_performance'))
    if pop_label:
        lines.append(f"Period of performance: {pop_label}")
    lines.append(f"Awarding agency: {_agency_label(data.get('awarding_agency'))}")
    if data.get("funding_agency"):
        lines.append(f"Funding agency: {_agency_label(data['funding_agency'])}")

    recipient = data.get("recipient") or {}
    recipient_location = recipient.get("location")
    lines.append(
        f"Recipient: {recipient.get('recipient_name', 'unknown')} ({_location_label(recipient_location)})"
    )
    district = _congressional_district_label(recipient_location)
    if district:
        lines.append(f"Recipient congressional district: {district}")
    lines.append(f"Place of performance: {_location_label(data.get('place_of_performance'))}")

    subaward_count = data.get("subaward_count") or 0
    if subaward_count:
        total_sub = data.get("total_subaward_amount")
        suffix = f", totaling ${total_sub:,.2f}" if total_sub else ""
        lines.append(f"Subawards: {subaward_count}{suffix}")

    contract_details = data.get("latest_transaction_contract_data") or {}
    if contract_details.get("extent_competed_description"):
        offers = contract_details.get("number_of_offers_received")
        suffix = f" ({offers} offers received)" if offers else ""
        lines.append(f"Competition: {contract_details['extent_competed_description']}{suffix}")
    if contract_details.get("type_of_contract_pricing_description"):
        lines.append(f"Contract pricing: {contract_details['type_of_contract_pricing_description']}")
    if contract_details.get("naics_description"):
        lines.append(f"NAICS: {contract_details['naics_description']}")
    if contract_details.get("product_or_service_description"):
        lines.append(f"Product/service: {contract_details['product_or_service_description']}")

    parent = data.get("parent_award")
    if parent:
        # generated_unique_award_id is the parent IDV's own internal_id -
        # without it, a follow-up get_award_details(include_child_orders=True)
        # call has no way to reach the parent VEHICLE, only this child
        # contract.
        vehicle_type = parent.get("type_of_idc_description") or parent.get("idv_type_description") or "contract vehicle"
        parent_internal_id = parent.get("generated_unique_award_id", "unknown")
        lines.append(
            f"Issued under parent IDV {parent.get('piid', 'unknown')} "
            f"({parent.get('agency_name', 'unknown agency')}, {vehicle_type}) "
            f"[internal_id: {parent_internal_id}]"
        )

    if data.get("category") == "idv":
        if child_order_rollup is not None:
            lines.append(
                f"Orders placed against this vehicle: {child_order_rollup.child_award_count} child awards "
                f"totaling ${child_order_rollup.child_award_total_obligation:,.2f} "
                f"(ceiling ${child_order_rollup.child_award_base_and_all_options_value:,.2f})"
            )
            if child_order_rollup.grandchild_award_count:
                lines.append(
                    f"Plus {child_order_rollup.grandchild_award_count} grandchild orders (orders placed "
                    f"against child IDVs nested under this one) totaling "
                    f"${child_order_rollup.grandchild_award_total_obligation:,.2f}"
                )
        else:
            lines.append(
                "(Note: this is a contract vehicle (IDV), not itself a spending transaction - the total "
                "obligated above reflects only the vehicle's own direct activity, not the orders placed "
                "against it. A real IDV can show $0 here while still being an active, heavily-used "
                "vehicle - do not present this figure as the total spent under this contract. Call "
                "again with include_child_orders=True for the real child-order rollup.)"
            )
    return "\n".join(lines)


# record_type 1 (Aggregate Record) and 3 (Non-Aggregate Record to an
# Individual Recipient with Redacted PII) per this project's own ingested
# Glossary ("Record Type" entry). record_type 1's recipient_name is
# literally "MULTIPLE RECIPIENTS", record_type 3's is "REDACTED DUE TO
# PII" - never shown to the model/user as the raw sentinel string; both
# get an explanatory label instead, and both get the state-only location
# (see _location_label's docstring for why).
_AGGREGATE_RECIPIENT_LABELS = {
    1: "Multiple recipients (aggregate award - individual identities aren't published, to protect personal privacy)",
    3: "Individual recipient (redacted - not published, to protect personal privacy)",
}


def _format_financial_assistance(data: dict[str, Any]) -> str:
    """Formatting for FinancialAssistanceResponse (grants/loans/direct
    payments/other assistance) - a genuinely different shape from
    ContractResponse/IDVResponse (fain/uri instead of piid, cfda_info
    instead of NAICS/PSC, no competition data, no parent-award linkage).

    total_subsidy_cost/total_loan_value and non_federal_funding/
    total_funding are shown only for their matching category (loans,
    grant) - conditioned on category, not on nullness, since these come
    back 0.0, not null, on every record they don't apply to (the
    award_id.md contract's own "null except for X" claim doesn't hold in
    practice).

    transaction_obligated_amount is deliberately never shown - it's a
    File C (Treasury-account-level financial reporting) sum, a genuinely
    different, separately-timed DATA Act submission from total_obligation's
    File D2 (award/transaction) source (only best-effort matched per
    usaspending-api's own C_to_D_Linkage.md, not guaranteed to agree) -
    same reasoning as excluding total_account_obligation for contracts/
    IDVs above. total_outlay is shown despite the same File-C caveat -
    see _format_contract_or_idv's docstring for why."""
    award_number = data.get("fain") or data.get("uri") or "unknown"
    lines = [f"{data.get('type_description', 'Unknown type')} ({award_number})"]
    if data.get("description"):
        lines.append(f"Description: {data['description']}")
    total_obligation = data.get("total_obligation") or 0
    category = data.get("category")
    # For a loan, total_obligation can be $0.00 while Loan value is
    # nonzero (e.g. a guaranteed loan) - a bare "$0.00" line right above a
    # large loan value reads as contradictory, so it's suppressed at zero
    # without asserting what total_obligation specifically measures for a
    # loan. Not hidden when genuinely nonzero.
    if not (category == "loans" and total_obligation == 0):
        lines.append(f"Total obligated: ${total_obligation:,.2f}")

    if category == "loans" and data.get("total_loan_value"):
        subsidy = data.get("total_subsidy_cost")
        suffix = f", subsidy cost ${subsidy:,.2f}" if subsidy else ""
        lines.append(f"Loan value: ${data['total_loan_value']:,.2f}{suffix}")
    elif category == "grant" and data.get("total_funding"):
        non_federal = data.get("non_federal_funding")
        suffix = f" (of which ${non_federal:,.2f} non-federal)" if non_federal else ""
        lines.append(f"Total funding: ${data['total_funding']:,.2f}{suffix}")

    outlay = data.get("total_outlay")
    if outlay is not None:
        lines.append(f"Outlayed: ${outlay:,.2f}")

    if data.get("date_signed"):
        lines.append(f"Date signed: {data['date_signed']}")
    pop_label = _format_period_of_performance(data.get('period_of_performance'))
    if pop_label:
        lines.append(f"Period of performance: {pop_label}")
    lines.append(f"Awarding agency: {_agency_label(data.get('awarding_agency'))}")
    if data.get("funding_agency"):
        lines.append(f"Funding agency: {_agency_label(data['funding_agency'])}")

    recipient = data.get("recipient") or {}
    recipient_location = recipient.get("location")
    record_type = data.get("record_type")
    if record_type in _AGGREGATE_RECIPIENT_LABELS:
        location = _location_label(recipient_location, full=False)
        lines.append(f"Recipient: {_AGGREGATE_RECIPIENT_LABELS[record_type]} ({location})")
    else:
        lines.append(
            f"Recipient: {recipient.get('recipient_name', 'unknown')} ({_location_label(recipient_location)})"
        )
        district = _congressional_district_label(recipient_location)
        if district:
            lines.append(f"Recipient congressional district: {district}")

    lines.append(f"Place of performance: {_location_label(data.get('place_of_performance'))}")

    subaward_count = data.get("subaward_count") or 0
    if subaward_count:
        total_sub = data.get("total_subaward_amount")
        suffix = f", totaling ${total_sub:,.2f}" if total_sub else ""
        lines.append(f"Subawards: {subaward_count}{suffix}")

    cfda_info = data.get("cfda_info") or []
    if cfda_info:
        programs = "; ".join(f"{c.get('cfda_number', '?')} {c.get('cfda_title') or ''}".strip() for c in cfda_info)
        lines.append(f"Assistance Listing(s): {programs}")

    return "\n".join(lines)


def _format_award_details(
    data: dict[str, Any], child_order_rollup: IDVAmountsResponse | None = None
) -> str:
    if data.get("category") in ("contract", "idv"):
        return _format_contract_or_idv(data, child_order_rollup)
    return _format_financial_assistance(data)


@beta_tool
def get_award_details(award_id: str, include_child_orders: bool = False) -> str:
    """Get full details about one specific award: a single contract, IDV (contract vehicle), grant, loan, or other financial assistance record. Returns description, dates, competition data (for contracts), recipient, funding breakdown, and (for contracts issued under an IDV) parent-vehicle info. Use this for "tell me more about this award/contract/grant" follow-up questions after search_awards — not for browsing or listing multiple awards, which search_awards already does.

    Args:
        award_id: The internal_id value shown alongside a search_awards result (e.g.
            "CONT_AWD_NSFDACS1219442_4900_-NONE-_-NONE-") — NOT the plain Award ID/PIID/FAIN
            shown next to it, which this endpoint doesn't accept. Only call this with an
            internal_id you already have — either from a prior search_awards result, or from a
            prior get_award_details call's own "Issued under parent IDV ... [internal_id: ...]"
            line, if the question is about the parent VEHICLE rather than the specific contract
            found by search_awards (e.g. "how much has been ordered under this IDV" — call
            get_award_details again on the parent's internal_id, with include_child_orders=True,
            rather than answering from the child contract's own total). Do not guess or
            construct an internal_id.
        include_child_orders: Set True only when the award is an IDV (a contract vehicle — BPA,
            GWAC, or multi-award IDC) AND the question is specifically about how much has been
            ordered under it (e.g. "how much has actually been spent under this contract
            vehicle"). False by default — it costs a second live API call and is meaningless for
            a plain contract/grant/loan/etc. An IDV's own total obligated (shown above regardless
            of this flag) reflects only the vehicle's own direct activity, not the orders placed
            against it — a real, active IDV can show $0 there. Setting this True fetches the
            actual rollup: how many child orders (and, for a nested vehicle, grandchild orders)
            exist and what they total.
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        data = get_award_details_raw(award_id)
    except USASpendingAPIError as e:
        logger.warning("get_award_details failed for %s: %s", award_id, e)
        return f"This query failed: {e}.{_unresolved_award_id_hint(award_id)}"

    child_order_rollup = None
    if include_child_orders and data.get("category") == "idv":
        try:
            child_order_rollup = get_idv_amounts_raw(award_id)
        except USASpendingAPIError as e:
            # Degrade gracefully rather than failing the whole call over an
            # enhancement fetch - _format_contract_or_idv falls back to its
            # existing caveat when child_order_rollup is None.
            logger.warning("get_idv_amounts failed for %s: %s", award_id, e)

    _record_tool_call(
        "get_award_details",
        data,
        {"award_id": award_id, "piid": data.get("piid") or data.get("fain") or data.get("uri")},
    )
    return _wrap_untrusted(_format_award_details(data, child_order_rollup))


@beta_tool
def get_award_subawards(award_id: str, limit: int = 10) -> str:
    """List the subawards issued under one specific prime award - the award-profile page's own Sub-Awards tab. Use this for "who did X subcontract this work to" or "what subawards has this award issued" about a SPECIFIC award already found via search_awards. For subawards across many awards (e.g. "subawards to Leidos" or "subawards issued by NSF"), use search_subawards instead - this tool only lists one award's own.

    Args:
        award_id: The internal_id shown alongside a search_awards result (or get_award_details' own
            award_id parameter) - the hash-style generated_unique_award_id, not the plain PIID/FAIN.
            Do not guess or construct one.
        limit: Max number of subawards to return, ranked by amount descending (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    client = _get_usaspending_client()
    try:
        response = client.get_award_subawards(award_id, limit=limit)
    except USASpendingAPIError as e:
        logger.warning("get_award_subawards failed for %s: %s", award_id, e)
        return f"This query failed: {e}."

    _record_tool_call("get_award_subawards", response, {"award_id": award_id})

    if not response.results:
        return f"No subawards found for award {award_id}."

    lines = [
        f"{s.subaward_number} — {s.recipient_name}: ${s.amount:,.2f} ({s.action_date}) - {s.description}"
        for s in response.results
    ]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
    return _wrap_untrusted("\n".join(lines) + note)


def _format_funding_row(row) -> str:
    account = row.federal_account or "unknown federal account"
    if row.account_title:
        account += f" ({row.account_title})"
    amount = row.transaction_obligated_amount or 0
    line = f"{account}: ${amount:,.2f} obligated"
    if row.gross_outlay_amount is not None:
        line += f", ${row.gross_outlay_amount:,.2f} outlayed"
    detail_bits = []
    if row.object_class_name or row.object_class:
        detail_bits.append(f"object class {row.object_class_name or row.object_class}")
    if row.program_activity_name:
        detail_bits.append(f"program activity {row.program_activity_name}")
    if row.disaster_emergency_fund_code:
        detail_bits.append(f"DEFC {row.disaster_emergency_fund_code}")
    if detail_bits:
        line += " (" + ", ".join(detail_bits) + ")"
    if row.reporting_fiscal_year:
        period = f"FY{row.reporting_fiscal_year}"
        if row.reporting_fiscal_quarter:
            period += f" Q{row.reporting_fiscal_quarter}"
        line += f" [{period}]"
    return line


@beta_tool
def get_award_funding_breakdown(award_id: str, limit: int = 10) -> str:
    """Get the Federal Account Funding breakdown for one specific award - which Treasury Account Symbol (TAS)/object class/program activity/Disaster Emergency Fund Code combinations actually funded it, and how much each contributed. This is the award-profile page's own "Federal Account Funding" tab. Use this for "which federal account(s) paid for this contract/grant" or "what TAS funded this award" questions about a SPECIFIC award already found via search_awards - not for browsing spending across many awards by account (no tool does that yet).

    This is a separate, later-timed data source (a Treasury account-level DATA Act "File C" submission) from get_award_details' own total_obligation (an award/transaction-level "File D2" figure) - the two are only best-effort linked, not guaranteed to sum to the same total. Always call get_award_details first for the award's own headline totals; call this only when the question is specifically about which federal account(s)/TAS funded it.

    Args:
        award_id: The internal_id shown alongside a search_awards result (or get_award_details' own
            award_id parameter) - the hash-style generated_unique_award_id, not the plain PIID/FAIN.
            Do not guess or construct one.
        limit: Max number of funding rows to return, ranked by reporting period descending (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_award_funding_breakdown_raw(award_id, limit=limit)
    except USASpendingAPIError as e:
        logger.warning("get_award_funding_breakdown failed for %s: %s", award_id, e)
        return f"This query failed: {e}."

    _record_tool_call("get_award_funding_breakdown", response, {"award_id": award_id})

    if not response.results:
        return f"No federal account funding data found for award {award_id}."

    lines = [_format_funding_row(row) for row in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
    return _wrap_untrusted("\n".join(lines) + note)


def _format_transaction_row(row) -> str:
    # federal_action_obligation is null for loans (face_value_loan_guarantee/
    # original_loan_subsidy_cost carry the amount there instead) - per
    # transactions.md, exactly one of the three is populated on any given row.
    if row.federal_action_obligation is not None:
        amount = f"${row.federal_action_obligation:,.2f}"
    elif row.face_value_loan_guarantee is not None:
        amount = f"${row.face_value_loan_guarantee:,.2f} (loan guarantee)"
    elif row.original_loan_subsidy_cost is not None:
        amount = f"${row.original_loan_subsidy_cost:,.2f} (loan subsidy cost)"
    else:
        amount = "amount unknown"
    mod = row.modification_number or "N/A"
    action = row.action_type_description or row.action_type or "unknown action"
    line = f"Mod {mod} ({row.action_date}): {amount} - {action}"
    if row.description:
        line += f" - {row.description}"
    return line


@beta_tool
def get_award_transaction_history(award_id: str, limit: int = 10) -> str:
    """List the individual transactions/modifications that built up to one specific award's current state - the award-profile page's own Transaction History tab (mod number, action date, action type, amount, description per row). Use this for "what modifications has this award had" or "show the transaction/mod history for this contract/grant" questions about a SPECIFIC award already found via search_awards - it does not aggregate or search across awards.

    Args:
        award_id: The internal_id shown alongside a search_awards result (or get_award_details' own
            award_id parameter) - the hash-style generated_unique_award_id, not the plain PIID/FAIN.
            Do not guess or construct one. Note: unlike get_award_details, a wrong/plain award_id
            does not error here - it silently returns no results, so an empty result on its own
            isn't proof the award genuinely has no transactions.
        limit: Max number of transactions to return, most recent action_date first (default 10).
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget
    try:
        response = get_award_transaction_history_raw(award_id, limit=limit)
    except USASpendingAPIError as e:
        logger.warning("get_award_transaction_history failed for %s: %s", award_id, e)
        return f"This query failed: {e}."

    _record_tool_call("get_award_transaction_history", response, {"award_id": award_id})

    if not response.results:
        return (
            f"No transactions found for award {award_id}.{_unresolved_award_id_hint(award_id)}"
        )

    lines = [_format_transaction_row(row) for row in response.results]
    has_next = response.page_metadata.hasNext if response.page_metadata else False
    note = _truncation_note(has_next, len(response.results))
    return _wrap_untrusted("\n".join(lines) + note)

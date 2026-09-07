"""Opt-in, real-billed verification of the shared filter layer
(award_amounts, recipient_search_text, location filters wired to all
three spending tools) and the search_awards sort fix.

Replays the exact findings from the audit that motivated this change,
end to end via ask() where practical, plus one direct client call to
settle the one open item flagged in tools.py's _amount_field_for_award_type
docstring (whether "Loan Value" is really the live field name for loan
award types, and whether sorting by it works). Not part of CI - run
manually:
    uv run python -m backend.app.agent.dev_tools.verify_shared_filters
"""
from __future__ import annotations

from backend.app.agent.orchestrator import ask
from backend.app.agent.tools import search_awards_raw


def case_search_awards_sorted_by_amount() -> None:
    print("\n=== search_awards now sorts largest-amount-first by default ===")
    question = "What were NSF's five biggest contracts in fiscal year 2023?"
    print(f"Question: {question}")
    result = ask(question)
    print(f"Answer: {result.answer_text}")
    print(
        "Finding this replays: pre-fix, this question returned contracts topping out "
        "around $7.2M while the true largest ($3.13B, NSFDACS1219442) never appeared. "
        "Check the answer above actually leads with a contract in the billions."
    )


def case_award_amount_filter() -> None:
    print("\n=== award_amounts filter: NSF contracts over $1 billion ===")
    results = search_awards_raw(
        "National Science Foundation", 2023, 2023, award_type="contracts", min_amount=1_000_000_000
    )
    print(f"Results: {results}")
    if len(results) == 1 and results[0].get("Award ID") == "NSFDACS1219442":
        print("OK: min_amount correctly isolated the one NSF contract over $1B.")
    else:
        print(f"FAIL: expected exactly one result (NSFDACS1219442), got {len(results)}.")


def case_recipient_search_text_filter() -> None:
    print("\n=== recipient_name filter, end to end ===")
    question = "How much has Leidos received from the Department of Defense in fiscal year 2023?"
    print(f"Question: {question}")
    result = ask(question)
    print(f"Answer: {result.answer_text}")
    print(
        "Pre-fix, this question either got an incomplete answer (no way to filter by "
        "recipient) or an honest decline. Check the answer above now cites a real, "
        "recipient-filtered figure."
    )


def case_location_filters_are_independent() -> None:
    print("\n=== performed_in_state vs recipient_in_state are different filters ===")
    # limit=1 isn't a strong enough check here: a shipbuilder like
    # Huntington Ingalls is both VA-headquartered AND performs work in VA,
    # so the single largest contract can legitimately satisfy both filters
    # at once without proving anything about whether they're wired
    # correctly. limit=10 + comparing the full ID lists is a real test.
    dod_contracts_performed_in_va = search_awards_raw(
        "Department of Defense", 2023, 2023, award_type="contracts",
        performed_in_state="Virginia", limit=10,
    )
    dod_contracts_recipient_in_va = search_awards_raw(
        "Department of Defense", 2023, 2023, award_type="contracts",
        recipient_in_state="Virginia", limit=10,
    )
    performed_ids = [r.get("Award ID") for r in dod_contracts_performed_in_va]
    recipient_ids = [r.get("Award ID") for r in dod_contracts_recipient_in_va]
    print(f"performed_in_state=Virginia, top 10 Award IDs: {performed_ids}")
    print(f"recipient_in_state=Virginia, top 10 Award IDs: {recipient_ids}")
    print(
        "Finding this replays: place_of_performance_locations vs. recipient_locations "
        "differ by ~$16B in aggregate for DoD/VA FY2023 (verified live 2026-09-06 via "
        "spending_over_time)."
    )
    if performed_ids == recipient_ids:
        print("FAIL: the two filters produced identical top-10 lists - likely wired to the same field.")
    else:
        print("OK: the two filters produce different result sets, as expected.")


def case_loan_amount_field_name() -> None:
    print("\n=== Open item: is 'Loan Value' really the field name for loan-type results? ===")
    # Small Business Administration issues direct/guaranteed loans - a
    # reliable agency to test the loan branch against.
    results = search_awards_raw("Small Business Administration", 2023, 2023, award_type="direct_loan", limit=3)
    print(f"Results: {results}")
    if results and "Loan Value" in results[0]:
        print("OK: 'Loan Value' is a real field on loan-type results, and sort didn't error.")
    elif not results:
        print("INCONCLUSIVE: no direct_loan results for SBA in FY2023 - try a different agency/year.")
    else:
        print(f"FAIL: 'Loan Value' not present on a real loan result - actual keys: {list(results[0].keys())}")


def main() -> None:
    case_search_awards_sorted_by_amount()
    case_award_amount_filter()
    case_recipient_search_text_filter()
    case_location_filters_are_independent()
    case_loan_amount_field_name()


if __name__ == "__main__":
    main()

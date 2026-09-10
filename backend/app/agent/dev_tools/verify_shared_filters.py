"""Opt-in, real-billed verification of the shared filter layer across all
three spending tools: award_amounts, recipient_search_text, location
filters, the search_awards sort fix, page_metadata.hasNext surfacing, and
date_type/new_awards_only (added 2026-09-07 after real UI usage found the
same award appearing under two different fiscal-year queries with an
identical total).

Replays the exact findings from the audit and from real UI usage that
motivated each fix, end to end via ask() where practical. Not part of
CI - run manually:
    uv run python -m backend.app.agent.dev_tools.verify_shared_filters
"""
from __future__ import annotations

from backend.app.agent.orchestrator import ask
from backend.app.agent.singletons import warm_up
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
    response = search_awards_raw(
        "National Science Foundation", 2023, 2023, award_type="contracts", min_amount=1_000_000_000
    )
    print(f"Results: {response.results}")
    if len(response.results) == 1 and response.results[0].get("Award ID") == "NSFDACS1219442":
        print("OK: min_amount correctly isolated the one NSF contract over $1B.")
    else:
        print(f"FAIL: expected exactly one result (NSFDACS1219442), got {len(response.results)}.")


def case_truncation_note_when_more_results_exist() -> None:
    print("\n=== hasNext -> a caveat note, not a silently truncated 'complete' list ===")
    # Real bug found live via the UI (2026-09-06): "NSF contracts worth
    # more than $10 million" with the default limit came back with
    # page_metadata.hasNext=true (only 5 of the real matches shown), and
    # the model presented that partial slice as the complete list. Same
    # min_amount, small limit, on purpose - should reproduce hasNext=true.
    response = search_awards_raw(
        "National Science Foundation", 2023, 2023, award_type="contracts",
        min_amount=10_000_000, limit=5,
    )
    has_next = response.page_metadata.hasNext if response.page_metadata else None
    print(f"Results returned: {len(response.results)}, page_metadata.hasNext: {has_next}")
    if has_next is True:
        print("OK: hasNext=true, as expected for this under-sized limit - the tool's formatted "
              "output should include a caveat (checked via search_awards's string output, not "
              "search_awards_raw's structured one - see the end-to-end case below).")
    elif has_next is False:
        print("NOTE: hasNext=false this time - either NSF has <=5 contracts over $10M this run, "
              "or something changed; not itself a failure, but re-run to confirm the caveat path.")
    else:
        print("FAIL: response.page_metadata is None - hasNext isn't being surfaced from the API at all.")

    # End-to-end: does the caveat actually reach the model's own answer text?
    question = "Show me NSF contracts worth more than $10 million in fiscal year 2023."
    print(f"\nQuestion: {question}")
    result = ask(question)
    print(f"Answer: {result.answer_text}")
    mentions_incompleteness = any(
        phrase in result.answer_text.lower()
        for phrase in ("more results", "not exhaustive", "not a complete", "additional", "may be more", "beyond")
    )
    if has_next and mentions_incompleteness:
        print("OK: the model's answer acknowledges the list may be incomplete.")
    elif has_next and not mentions_incompleteness:
        print("FAIL: hasNext was true but the model's answer doesn't flag incompleteness anywhere.")


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
    performed_ids = [r.get("Award ID") for r in dod_contracts_performed_in_va.results]
    recipient_ids = [r.get("Award ID") for r in dod_contracts_recipient_in_va.results]
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
    response = search_awards_raw("Small Business Administration", 2023, 2023, award_type="direct_loan", limit=3)
    print(f"Results: {response.results}")
    if response.results and "Loan Value" in response.results[0]:
        print("OK: 'Loan Value' is a real field on loan-type results, and sort didn't error.")
    elif not response.results:
        print("INCONCLUSIVE: no direct_loan results for SBA in FY2023 - try a different agency/year.")
    else:
        print(f"FAIL: 'Loan Value' not present on a real loan result - actual keys: {list(response.results[0].keys())}")


def case_new_awards_only_eliminates_cross_fiscal_year_duplication() -> None:
    print("\n=== date_type='new_awards_only' fixes the cross-FY duplication bug ===")
    # Real bug found live via the UI (2026-09-07): the same NSF contract
    # (e.g. Accenture Federal Services, $45,032,222.26) appeared under both
    # a FY2023 "contracts over $10M" query and a FY2024 "$10-50M" query,
    # same Award ID, same total - because the default date_type
    # (action_date) matches on ANY transaction in the window, and Award
    # Amount is the award's cumulative total, not period-scoped.
    default_fy23 = search_awards_raw(
        "National Science Foundation", 2023, 2023, award_type="contracts",
        min_amount=10_000_000, limit=25,
    )
    default_fy24 = search_awards_raw(
        "National Science Foundation", 2024, 2024, award_type="contracts",
        min_amount=10_000_000, max_amount=50_000_000, limit=25,
    )
    default_overlap = {r["Award ID"] for r in default_fy23.results} & {r["Award ID"] for r in default_fy24.results}
    print(f"Default date_type overlap between FY23/FY24 (expected: some, reproducing the original bug): "
          f"{len(default_overlap)}")

    new_only_fy23 = search_awards_raw(
        "National Science Foundation", 2023, 2023, award_type="contracts",
        min_amount=10_000_000, date_type="new_awards_only", limit=25,
    )
    new_only_fy24 = search_awards_raw(
        "National Science Foundation", 2024, 2024, award_type="contracts",
        min_amount=10_000_000, max_amount=50_000_000, date_type="new_awards_only", limit=25,
    )
    new_only_overlap = {r["Award ID"] for r in new_only_fy23.results} & {r["Award ID"] for r in new_only_fy24.results}
    print(f"new_awards_only overlap between FY23/FY24 (expected: 0): {len(new_only_overlap)}")

    if default_overlap and not new_only_overlap:
        print("OK: default date_type reproduces the original bug, new_awards_only eliminates it.")
    elif not new_only_overlap:
        print("PARTIAL: new_awards_only has zero overlap (good), but default also had zero this run - "
              "re-run to reconfirm the default reproduces the bug (result set changes over time).")
    else:
        print(f"FAIL: new_awards_only still overlaps: {new_only_overlap}")


def main() -> None:
    # Required for AGENT_ENGINE=langgraph (_get_conversation_graph() asserts
    # warm_up() already ran); harmless no-op cost for the legacy path.
    warm_up()
    case_search_awards_sorted_by_amount()
    case_award_amount_filter()
    case_truncation_note_when_more_results_exist()
    case_recipient_search_text_filter()
    case_location_filters_are_independent()
    case_loan_amount_field_name()
    case_new_awards_only_eliminates_cross_fiscal_year_duplication()


if __name__ == "__main__":
    main()

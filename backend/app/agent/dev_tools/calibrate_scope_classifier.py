"""Calibrate the scope classifier (_is_in_scope) against a hand-reviewed
labeled set, instead of judging it from a handful of hand-picked examples
the way the "what is NSF's mission" / "what is a BOA" gaps were first
found.

Two things this measures that a single pass over the labeled set can't:

1. Stability. The classifier has no temperature control (sampling
   parameters are deprecated/rejected outright by the current API) and is
   confirmed non-deterministic - "what is a BOA" flips True/False across
   identical calls. Each question is called N_REPEATS times so accuracy
   and agreement (how often the N calls agree with each other, independent
   of correctness) can both be measured, instead of one noisy sample.

2. Whether majority-vote or RAG-context augmentation actually help, on
   real measured numbers instead of reasoning about it. Three variants,
   compared against the same labeled set:
     - baseline: today's classifier, bare question text
     - majority_vote: same calls as baseline, majority of N_REPEATS wins
       (no extra calls needed - just a different aggregation of the same
       samples)
     - rag_augmented: the question plus the single top retrieved passage
       (Guide + Glossary), with the prompt explicitly told a weak/no match
       does NOT imply out-of-scope - live-data questions legitimately have
       no match in this corpus at all, and the risk of leaning on retrieval
       quality is exactly that it could make those questions worse instead
       of fixing the definitional-term gap it's meant to close.

Usage:
  python -m backend.app.agent.dev_tools.calibrate_scope_classifier
"""
from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from backend.app.agent.singletons import MODEL, _get_client
from backend.app.retrieval.hybrid import HybridRetriever

LABELED_SET_PATH = Path(__file__).parent / "scope_classifier_labeled_set.json"
N_REPEATS = 5
MAX_WORKERS = 8

BASELINE_PROMPT = (
    "You classify whether a user question is in scope for a USASpending.gov "
    "assistant: federal spending, budgets, obligations/outlays, contracts, "
    "grants/financial assistance, awards, recipients, federal agencies, or "
    "USASpending.gov data/fields/API concepts. Respond with only YES or NO, "
    "nothing else."
)

RAG_AUGMENTED_PROMPT = (
    "You classify whether a user question is in scope for a USASpending.gov "
    "assistant: federal spending, budgets, obligations/outlays, contracts, "
    "grants/financial assistance, awards, recipients, federal agencies, or "
    "USASpending.gov data/fields/API concepts. You are given the single most "
    "relevant passage a retrieval system found for this question, which may "
    "or may not actually be relevant — a weak or unrelated passage does NOT "
    "mean the question is out of scope, since many in-scope questions (e.g. "
    "live spending-data lookups) have no good match in this retrieval corpus "
    "at all. Use the passage only as supporting evidence when it looks "
    "genuinely on-topic; ignore it if it looks irrelevant. Respond with only "
    "YES or NO, nothing else."
)


def load_labeled_set() -> list[dict]:
    with LABELED_SET_PATH.open(encoding="utf-8") as fh:
        data = json.load(fh)
    entries = []
    for category, items in data.items():
        if category == "_notes":
            continue
        for item in items:
            entries.append({**item, "category": category})
    return entries


def classify_baseline(question: str) -> bool:
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=BASELINE_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")


def classify_rag_augmented(question: str, passage: str | None, score: float | None) -> bool:
    if passage is None:
        context = "No relevant passage was found for this question."
    else:
        context = f"Most relevant passage found (rerank score {score:.2f}):\n{passage}"
    response = _get_client().messages.create(
        model=MODEL,
        max_tokens=5,
        system=RAG_AUGMENTED_PROMPT,
        messages=[{"role": "user", "content": f"Question: {question}\n\n{context}"}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return text.strip().upper().startswith("YES")


def get_top_passage(retriever: HybridRetriever, question: str) -> tuple[str | None, float | None]:
    results = retriever.retrieve(question, top_k=1)
    if not results:
        return None, None
    top = results[0]
    return top["text"], top["rerank_score"]


def run_repeated(fn, n: int) -> list[bool]:
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda _: fn(), range(n)))


def evaluate_entry(entry: dict, retriever: HybridRetriever) -> dict:
    question = entry["question"]
    expected = entry["label"] == "in_scope"

    baseline_calls = run_repeated(lambda: classify_baseline(question), N_REPEATS)
    passage, score = get_top_passage(retriever, question)
    rag_calls = run_repeated(lambda: classify_rag_augmented(question, passage, score), N_REPEATS)

    def summarize(calls: list[bool]) -> dict:
        majority = sum(calls) > len(calls) / 2
        single_shot_correct = sum(1 for c in calls if c == expected) / len(calls)
        agreement = max(sum(calls), len(calls) - sum(calls)) / len(calls)
        return {
            "calls": calls,
            "majority_vote": majority,
            "majority_correct": majority == expected,
            "single_shot_accuracy": single_shot_correct,
            "agreement_rate": agreement,
        }

    return {
        "question": question,
        "category": entry["category"],
        "expected": expected,
        "top_passage_score": score,
        "baseline": summarize(baseline_calls),
        "rag": summarize(rag_calls),
    }


def print_report(results: list[dict]) -> None:
    print(f"\n{'category':<38} {'base 1shot':>10} {'base maj':>9} {'rag 1shot':>10} {'rag maj':>8} {'n':>4}")
    print("-" * 85)

    by_category = defaultdict(list)
    for r in results:
        by_category[r["category"]].append(r)

    def agg(rows: list[dict], variant: str, key: str) -> float:
        return sum(row[variant][key] for row in rows) / len(rows)

    for category, rows in by_category.items():
        print(
            f"{category:<38} "
            f"{agg(rows, 'baseline', 'single_shot_accuracy'):>10.1%} "
            f"{sum(row['baseline']['majority_correct'] for row in rows) / len(rows):>9.1%} "
            f"{agg(rows, 'rag', 'single_shot_accuracy'):>10.1%} "
            f"{sum(row['rag']['majority_correct'] for row in rows) / len(rows):>8.1%} "
            f"{len(rows):>4}"
        )

    print("-" * 85)
    print(
        f"{'OVERALL':<38} "
        f"{agg(results, 'baseline', 'single_shot_accuracy'):>10.1%} "
        f"{sum(r['baseline']['majority_correct'] for r in results) / len(results):>9.1%} "
        f"{agg(results, 'rag', 'single_shot_accuracy'):>10.1%} "
        f"{sum(r['rag']['majority_correct'] for r in results) / len(results):>8.1%} "
        f"{len(results):>4}"
    )

    avg_agreement = sum(r["baseline"]["agreement_rate"] for r in results) / len(results)
    print(f"\nMean baseline agreement rate across {N_REPEATS} repeats (flakiness indicator): {avg_agreement:.1%}")
    print("(100% = every repeat agreed; 60% is the floor for N=5, i.e. a 3/2 split)")

    print("\nQuestions where baseline is flaky (not 100% agreement):")
    for r in results:
        if r["baseline"]["agreement_rate"] < 1.0:
            print(
                f"  [{r['category']}] {r['question']!r} - baseline calls: {r['baseline']['calls']}, "
                f"rag calls: {r['rag']['calls']}"
            )

    print("\nQuestions where baseline majority-vote is wrong:")
    for r in results:
        if not r["baseline"]["majority_correct"]:
            print(f"  [{r['category']}] {r['question']!r} - expected {r['expected']}, baseline calls: {r['baseline']['calls']}")

    print("\nQuestions where rag majority-vote is wrong:")
    for r in results:
        if not r["rag"]["majority_correct"]:
            print(f"  [{r['category']}] {r['question']!r} - expected {r['expected']}, rag calls: {r['rag']['calls']}")


def main() -> None:
    entries = load_labeled_set()
    print(f"Loaded {len(entries)} labeled questions. Running {N_REPEATS} repeats x 2 variants each "
          f"({len(entries) * N_REPEATS * 2} total classifier calls)...")

    retriever = HybridRetriever()
    results = []
    for i, entry in enumerate(entries, start=1):
        print(f"  [{i}/{len(entries)}] {entry['question'][:70]!r}")
        results.append(evaluate_entry(entry, retriever))

    print_report(results)


if __name__ == "__main__":
    main()

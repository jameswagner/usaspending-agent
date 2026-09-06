"""Calibrate RERANK_CONFIDENCE_THRESHOLD against a generated labeled set,
instead of the eyeballed -5.0 picked from ~8 examples in sanity_check.py.

Generates two sets of questions:
  - positive: paraphrased questions that real guide chunks directly answer
  - negative: a mix of (a) completely off-domain questions and (b) on-domain
    questions the static guide can't answer (needs a live number/fact) —
    the harder, more important case, since search_guide's threshold should
    say "nothing relevant" for these even though the topic is legitimate.

Runs each question through HybridRetriever, records the top rerank score,
and finds the threshold that best separates the two labeled groups.

Usage:
  python -m backend.app.retrieval.dev_tools.calibrate_threshold
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from dotenv import load_dotenv

from backend.app.agent.singletons import _get_client
from backend.app.retrieval.hybrid import HybridRetriever

load_dotenv()

CHUNKS_PATH = Path("data/chunks/analysts_guide_chunks.jsonl")
MODEL = "claude-haiku-4-5"
N_POSITIVE = 20
N_NEGATIVE = 20


def load_chunks() -> list[dict]:
    with CHUNKS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def generate_positive_questions(chunks: list[dict], n: int) -> list[str]:
    """One natural question per sampled chunk, that the chunk directly answers."""
    client = _get_client()
    sample = random.sample(chunks, n)
    questions = []
    for c in sample:
        response = client.messages.create(
            model=MODEL,
            max_tokens=100,
            system=(
                "Given a passage from a federal spending data guide, write ONE "
                "natural question a user might ask that this passage directly "
                "and fully answers. Respond with only the question text, "
                "nothing else — no numbering, no quotes."
            ),
            messages=[{"role": "user", "content": c["text"]}],
        )
        q = next((b.text for b in response.content if b.type == "text"), "").strip()
        if q:
            questions.append(q)
    return questions


def generate_negative_questions(n: int) -> list[str]:
    """Half off-domain entirely, half on-domain but needing live data the
    static guide can't provide — the harder, more realistic negative case.
    """
    client = _get_client()
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=(
            f"Generate exactly {n} questions for testing a retrieval system's "
            "false-positive rate, one per line, no numbering.\n\n"
            f"The first {n // 2} should be completely unrelated to US federal "
            "spending, budgets, or government agencies (everyday topics: "
            "cooking, sports, travel, other countries, etc.).\n\n"
            f"The last {n - n // 2} should be ABOUT US federal spending or "
            "USASpending.gov, but ask for a specific live number, current "
            "fact, or named-entity lookup that a static conceptual "
            "glossary/FAQ document could not answer — e.g. a specific "
            "agency's dollar amount for a specific year, not a definition "
            "of a term like 'obligation' or 'prime award'."
        ),
        messages=[{"role": "user", "content": f"Generate the {n} questions now."}],
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    return [line.strip() for line in text.splitlines() if line.strip()]


def score_questions(retriever: HybridRetriever, questions: list[str], label: str) -> list[tuple[float, str, str]]:
    scored = []
    for q in questions:
        results = retriever.retrieve(q, top_k=1)
        top_score = results[0]["rerank_score"] if results else -999.0
        scored.append((top_score, label, q))
    return scored


def best_threshold(scored: list[tuple[float, str, str]]) -> tuple[float, float]:
    """Try every midpoint between consecutive scores; return the one that
    maximizes accuracy (positive above threshold, negative at/below it).
    """
    scores = sorted(s for s, _, _ in scored)
    candidates = [scores[0] - 1] + [
        (scores[i] + scores[i + 1]) / 2 for i in range(len(scores) - 1)
    ]

    best_t, best_acc = candidates[0], -1.0
    for t in candidates:
        correct = sum(
            1
            for score, label, _ in scored
            if (score > t and label == "positive") or (score <= t and label == "negative")
        )
        acc = correct / len(scored)
        if acc > best_acc:
            best_t, best_acc = t, acc
    return best_t, best_acc


def main():
    chunks = load_chunks()
    print(f"Generating {N_POSITIVE} positive and {N_NEGATIVE} negative questions...")
    positives = generate_positive_questions(chunks, N_POSITIVE)
    negatives = generate_negative_questions(N_NEGATIVE)

    retriever = HybridRetriever()
    print("Scoring...")
    scored = score_questions(retriever, positives, "positive") + score_questions(retriever, negatives, "negative")
    scored.sort(key=lambda x: x[0])

    print(f"\n{'score':>8}  {'label':<9}  question")
    print("-" * 80)
    for score, label, q in scored:
        print(f"{score:8.2f}  {label:<9}  {q[:65]}")

    threshold, accuracy = best_threshold(scored)
    print(f"\nBest-separating threshold: {threshold:.2f} (accuracy on this labeled set: {accuracy:.1%})")
    print("Current RERANK_CONFIDENCE_THRESHOLD: -2.0")


if __name__ == "__main__":
    main()

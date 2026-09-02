"""Run a curated batch of example queries through the hybrid retriever.

A quick eyeball check that results look sensible across different query
styles (exact terms, acronyms, typos, paraphrases, comparisons, and
out-of-scope questions), without reloading the models on every query like
the one-shot hybrid.py CLI does.

Usage:
  python -m backend.app.retrieval.sanity_check
"""
from __future__ import annotations

from backend.app.retrieval.hybrid import HybridRetriever

# (category, query) pairs, each probing a different retrieval behavior.
QUERIES = [
    ("verbatim from doc", "What are sub-tier agencies?"),
    ("acronym only", "What is IDV"),
    ("acronym only", "NAICS"),
    ("misspelled", "recepient identifer feilds"),
    ("paraphrase, no shared vocab", "how do I know who got the money"),
    ("comparison", "grant versus contract"),
    ("ambiguous single word", "loan"),
    ("out of scope", "what's the capital of France"),
]


def run(top_k: int = 3) -> None:
    retriever = HybridRetriever()

    for category, query in QUERIES:
        print(f"\n{'=' * 80}")
        print(f"[{category}] query: {query!r}")
        print("=" * 80)

        results = retriever.retrieve(query, top_k=top_k)
        if not results:
            print("  (no candidates found)")
            continue

        for i, r in enumerate(results, start=1):
            dense = f"dense_rank={r['dense_rank']}" if "dense_rank" in r else "dense_rank=-"
            sparse = f"sparse_rank={r['sparse_rank']}" if "sparse_rank" in r else "sparse_rank=-"
            preview = r["text"][:150].replace("\n", " ")
            print(f"  {i}. rerank={r['rerank_score']:.2f}  {dense}  {sparse}  page={r['page_start']}")
            print(f"     {preview}...")


if __name__ == "__main__":
    run()

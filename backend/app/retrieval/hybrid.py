"""Hybrid dense + BM25 retriever with cross-encoder reranking.

Runs the Chroma (dense) and Whoosh (BM25) searches independently, merges
the candidate chunks by id, then rescores the merged set with a
cross-encoder for the final ranking.

Usage:
  python -m backend.app.retrieval.hybrid --query "What is a prime award?"
"""
from __future__ import annotations

import os

import chromadb
from dotenv import load_dotenv
from langsmith import traceable
from sentence_transformers import CrossEncoder, SentenceTransformer
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Loaded here (not just in main.py) so LANGSMITH_* and index-path env vars
# also take effect when this module is run standalone (hybrid.py's own CLI,
# sanity_check.py) rather than only inside the FastAPI server.
load_dotenv()

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
WHOOSH_INDEX_DIR = os.environ.get("WHOOSH_INDEX_DIR", "./data/whoosh")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = "analysts_guide"


def merge_candidates(dense: dict[str, dict], sparse: dict[str, dict]) -> list[dict]:
    """Union dense and sparse candidates by chunk id.

    A chunk found by only one retriever keeps just that retriever's
    rank/score fields; a chunk found by both keeps both (sparse fields
    merged onto the dense dict, since either side already carries the full
    chunk text/metadata). Pulled out as a standalone function so the merge
    logic is unit-testable without loading the embedding/cross-encoder
    models or touching Chroma/Whoosh.
    """
    merged: dict[str, dict] = {}
    for chunk_id, c in dense.items():
        merged[chunk_id] = c
    for chunk_id, c in sparse.items():
        if chunk_id in merged:
            merged[chunk_id]["sparse_rank"] = c["sparse_rank"]
            merged[chunk_id]["sparse_score"] = c["sparse_score"]
        else:
            merged[chunk_id] = c
    return list(merged.values())


class HybridRetriever:
    def __init__(self, dense_k: int = 10, sparse_k: int = 10):
        self.dense_k = dense_k
        self.sparse_k = sparse_k

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)
        self.whoosh_ix = open_dir(WHOOSH_INDEX_DIR)

    @traceable(run_type="retriever", name="dense_search_chroma")
    def _dense_search(self, query: str) -> dict[str, dict]:
        embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=self.dense_k)

        candidates = {}
        for rank, (chunk_id, text, meta, distance) in enumerate(
            zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ),
            start=1,
        ):
            candidates[chunk_id] = {
                "id": chunk_id,
                "text": text,
                **meta,
                "dense_rank": rank,
                "dense_distance": distance,
            }
        return candidates

    @traceable(run_type="retriever", name="sparse_search_bm25")
    def _sparse_search(self, query: str) -> dict[str, dict]:
        candidates = {}
        with self.whoosh_ix.searcher() as searcher:
            parser = QueryParser("text", self.whoosh_ix.schema)
            parsed_query = parser.parse(query)
            for rank, r in enumerate(searcher.search(parsed_query, limit=self.sparse_k), start=1):
                candidates[r["id"]] = {
                    "id": r["id"],
                    "text": r["text"],
                    "source": r["source"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "paragraph_index": r["paragraph_index"],
                    "sparse_rank": rank,
                    "sparse_score": r.score,
                }
        return candidates

    @traceable(run_type="chain", name="cross_encoder_rerank")
    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates

    @traceable(run_type="chain", name="hybrid_retrieve")
    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        candidates = merge_candidates(self._dense_search(query), self._sparse_search(query))

        if not candidates:
            return []

        ranked = self._rerank(query, candidates)
        return ranked[:top_k]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for i, r in enumerate(results, start=1):
        dense_str = (
            f"dense_rank={r['dense_rank']} dense_distance={r['dense_distance']:.4f}"
            if "dense_rank" in r
            else "dense_rank=- (not in dense top-k)"
        )
        sparse_str = (
            f"sparse_rank={r['sparse_rank']} sparse_score={r['sparse_score']:.4f}"
            if "sparse_rank" in r
            else "sparse_rank=- (not in sparse top-k)"
        )
        print(f"--- final rank {i} | rerank_score={r['rerank_score']:.4f} | page={r['page_start']} ---")
        print(f"    {dense_str}")
        print(f"    {sparse_str}")
        print(r["text"][:300])
        print()


if __name__ == "__main__":
    main()

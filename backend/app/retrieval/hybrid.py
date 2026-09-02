"""Hybrid dense + BM25 retriever with cross-encoder reranking.

Runs the Chroma (dense) and Whoosh (BM25) searches independently, merges
the candidate chunks by id, then rescores the merged set with a
cross-encoder for the final ranking.

Usage:
  python -m backend.app.retrieval.hybrid --query "What is a prime award?"
"""
from __future__ import annotations

import os
from typing import Dict, List

import chromadb
from sentence_transformers import CrossEncoder, SentenceTransformer
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
WHOOSH_INDEX_DIR = os.environ.get("WHOOSH_INDEX_DIR", "./data/whoosh")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
COLLECTION_NAME = "analysts_guide"


class HybridRetriever:
    def __init__(self, dense_k: int = 10, sparse_k: int = 10):
        self.dense_k = dense_k
        self.sparse_k = sparse_k

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

        chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = chroma_client.get_collection(COLLECTION_NAME)
        self.whoosh_ix = open_dir(WHOOSH_INDEX_DIR)

    def _dense_search(self, query: str) -> Dict[str, dict]:
        embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=embedding, n_results=self.dense_k)

        candidates = {}
        for chunk_id, text, meta in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0]
        ):
            candidates[chunk_id] = {"id": chunk_id, "text": text, **meta}
        return candidates

    def _sparse_search(self, query: str) -> Dict[str, dict]:
        candidates = {}
        with self.whoosh_ix.searcher() as searcher:
            parser = QueryParser("text", self.whoosh_ix.schema)
            parsed_query = parser.parse(query)
            for r in searcher.search(parsed_query, limit=self.sparse_k):
                candidates[r["id"]] = {
                    "id": r["id"],
                    "text": r["text"],
                    "source": r["source"],
                    "page_start": r["page_start"],
                    "page_end": r["page_end"],
                    "paragraph_index": r["paragraph_index"],
                }
        return candidates

    def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        merged: Dict[str, dict] = {}
        merged.update(self._dense_search(query))
        merged.update(self._sparse_search(query))
        candidates = list(merged.values())

        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self.cross_encoder.predict(pairs)
        for c, score in zip(candidates, scores):
            c["rerank_score"] = float(score)

        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        return candidates[:top_k]


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    retriever = HybridRetriever()
    results = retriever.retrieve(args.query, top_k=args.top_k)

    for i, r in enumerate(results, start=1):
        print(f"--- rank {i} (rerank_score={r['rerank_score']:.4f}, page={r['page_start']}) ---")
        print(r["text"][:300])
        print()


if __name__ == "__main__":
    main()

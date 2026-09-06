"""Embed chunks and load them into a persistent Chroma collection.

Usage:
  python -m backend.app.retrieval.pipeline.vector_index \\
      --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "analysts_guide"


def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_index(chunks_paths: list[Path]) -> None:
    # Every source's chunks go into one collection - search_guide searches
    # across all of them together, not per-source indexes - so this takes
    # multiple chunk files and rebuilds the whole collection from all of
    # them in one shot, rather than one file appending to the last run
    # (which "drop and recreate" would otherwise wipe on the next source's
    # turn).
    chunks = [c for p in chunks_paths for c in load_chunks(p)]

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Drop and recreate the collection each run so re-ingesting doesn't
    # leave stale or duplicate chunks behind.
    existing = {c.name for c in client.list_collections()}
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["text"] for c in chunks],
        metadatas=[
            {
                "source": c["source"],
                "page_start": c["page_start"],
                "page_end": c["page_end"],
                "paragraph_index": c["paragraph_index"],
                # Glossary-only fields - Chroma metadata values must be
                # scalar (str/int/float/bool), so related_slugs (a list)
                # is JSON-encoded; empty/"" for Guide chunks, which don't
                # have these.
                "term": c.get("term", ""),
                "slug": c.get("slug", ""),
                "related_slugs": json.dumps(c.get("related_slugs", [])),
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}' at {CHROMA_DB_DIR}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        nargs="+",
        default=["data/chunks/analysts_guide_chunks.jsonl", "data/chunks/glossary_chunks.jsonl"],
    )
    args = parser.parse_args()

    build_index([Path(p) for p in args.chunks])


if __name__ == "__main__":
    main()

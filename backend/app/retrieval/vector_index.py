"""Embed chunks and load them into a persistent Chroma collection.

Usage:
  python -m backend.app.retrieval.vector_index --chunks data/chunks/analysts_guide_chunks.jsonl
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DB_DIR = os.environ.get("CHROMA_DB_DIR", "./data/chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "analysts_guide"


def load_chunks(chunks_path: Path) -> List[dict]:
    with chunks_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_index(chunks_path: Path) -> None:
    chunks = load_chunks(chunks_path)

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)

    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    # Drop and recreate the collection each run so re-ingesting the same
    # source doesn't leave stale or duplicate chunks behind.
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
            }
            for c in chunks
        ],
    )
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}' at {CHROMA_DB_DIR}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="data/chunks/analysts_guide_chunks.jsonl")
    args = parser.parse_args()

    build_index(Path(args.chunks))


if __name__ == "__main__":
    main()

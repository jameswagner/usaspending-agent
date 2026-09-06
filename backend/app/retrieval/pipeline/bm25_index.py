"""Build a Whoosh BM25 index over chunks for sparse keyword search.

Usage:
  python -m backend.app.retrieval.pipeline.bm25_index \\
      --chunks data/chunks/analysts_guide_chunks.jsonl data/chunks/glossary_chunks.jsonl
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from whoosh.fields import ID, NUMERIC, STORED, TEXT, Schema
from whoosh.index import create_in

WHOOSH_INDEX_DIR = os.environ.get("WHOOSH_INDEX_DIR", "./data/whoosh")

SCHEMA = Schema(
    id=ID(stored=True, unique=True),
    text=TEXT(stored=True),
    source=STORED,
    page_start=NUMERIC(stored=True),
    page_end=STORED,
    paragraph_index=STORED,
    # Glossary-only fields, stored but not separately searchable (the
    # term is already embedded in `text`, same as page_start etc. above
    # for the Guide) - "" / [] for Guide chunks, which don't have these.
    term=STORED,
    slug=STORED,
    related_slugs=STORED,
)


def load_chunks(chunks_path: Path) -> list[dict]:
    with chunks_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def build_index(chunks_paths: list[Path]) -> None:
    # Same reasoning as vector_index.py: every source's chunks go into one
    # index, rebuilt from all of them together in one shot.
    chunks = [c for p in chunks_paths for c in load_chunks(p)]

    index_dir = Path(WHOOSH_INDEX_DIR)
    # Rebuild from scratch each run so re-ingesting doesn't leave stale
    # documents behind.
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    ix = create_in(index_dir, SCHEMA)
    writer = ix.writer()
    for c in chunks:
        writer.add_document(
            id=c["id"],
            text=c["text"],
            source=c["source"],
            page_start=c["page_start"],
            page_end=c["page_end"],
            paragraph_index=c["paragraph_index"],
            term=c.get("term", ""),
            slug=c.get("slug", ""),
            # JSON-encoded, matching vector_index.py - Chroma metadata
            # can't hold a list, so this keeps the type consistent
            # regardless of which retriever (dense or sparse) surfaces a
            # given chunk, rather than a list here and a string there.
            related_slugs=json.dumps(c.get("related_slugs", [])),
        )
    writer.commit()
    print(f"Indexed {len(chunks)} chunks into Whoosh index at {index_dir}")


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

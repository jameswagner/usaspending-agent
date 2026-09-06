"""Ingest the live USASpending Glossary API into chunk records.

Unlike the Analyst's Guide (a PDF needing Q&A-boundary and character-budget
chunking), the glossary API already returns clean structured JSON, and
each entry is naturally the right size and shape for one chunk - one term,
one chunk, no further splitting.

Usage:
  python -m backend.app.retrieval.pipeline.ingest_glossary
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

GLOSSARY_URL = "https://api.usaspending.gov/api/v2/references/glossary/"
SOURCE_NAME = "USASpending Glossary"

# Cross-references to other glossary terms live inside the free-text
# `resources` field as markdown links to "?glossary=<slug>", mixed in with
# plain external URLs (e.g. a GAO PDF) that this deliberately ignores -
# verified against the live response, not assumed: `resources` is
# unstructured markdown text, not a structured list, e.g.
# "See also:\n\n- [Federal Account](?glossary=federal-account)".
GLOSSARY_LINK_RE = re.compile(r"\?glossary=([a-z0-9-]+)")


@dataclass
class GlossaryChunk:
    id: str
    source: str
    page_start: int
    page_end: int
    paragraph_index: int
    text: str
    term: str
    slug: str
    related_slugs: list[str]


def fetch_glossary() -> list[dict]:
    """Single request - verified live that the endpoint returns all ~151
    entries in one page (page_metadata.hasNext is False, count == len(
    results)), so no pagination loop is needed here."""
    resp = requests.get(GLOSSARY_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()["results"]


def extract_related_slugs(resources: str | None) -> list[str]:
    if not resources:
        return []
    return GLOSSARY_LINK_RE.findall(resources)


def entry_to_chunk(entry: dict) -> GlossaryChunk:
    term = entry["term"]
    slug = entry["slug"]
    plain = entry["plain"]
    official = entry.get("official")
    data_act_term = entry.get("data_act_term")

    heading = term
    if data_act_term and data_act_term != term:
        heading = f"{term} (DATA Act term: {data_act_term})"

    text_parts = [heading, plain]
    # official is real additional content when present, not metadata - but
    # roughly a third of entries repeat plain verbatim in official, which
    # would just duplicate the chunk's text for no benefit.
    if official and official != plain:
        text_parts.append(f"Official definition: {official}")
    text = "\n\n".join(text_parts)

    return GlossaryChunk(
        id=f"Glossary_{slug}",
        source=SOURCE_NAME,
        # Not a paginated document like the Analyst's Guide - these three
        # fields have no real meaning for a glossary entry. Kept as 0
        # placeholders (rather than making them optional everywhere) so
        # the shared Chroma/Whoosh schema built around the Guide's
        # page-based chunks doesn't need restructuring. Citations use
        # `term`/`slug` instead of these for glossary chunks - see
        # orchestrator.py's citation-building and Citation's `term` field.
        page_start=0,
        page_end=0,
        paragraph_index=0,
        text=text,
        term=term,
        slug=slug,
        related_slugs=extract_related_slugs(entry.get("resources")),
    )


def ingest_glossary_to_chunks(out_path: Path) -> list[GlossaryChunk]:
    entries = fetch_glossary()
    chunks = [entry_to_chunk(e) for e in entries]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return chunks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/chunks/glossary_chunks.jsonl")
    args = parser.parse_args()

    chunks = ingest_glossary_to_chunks(Path(args.out))
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()

"""Ingest the SAM.gov Assistance Listings (CFDA) bulk CSV into chunk
records, for semantic lookup - the cfda_program counterpart to
ingest_naics.py/ingest_psc.py. Third of three for #6: resolve_cfda_program
(not yet wired up) resolves a plain-English description to a program
number for search_awards/get_spending_by_category's cfda_program filter.

One clean source file, no real wrinkles (unlike NAICS's range codes or
PSC's retired-row duplicates) - confirmed against the real file, not
assumed: 2,868 rows, zero duplicate Program Numbers, every row has a
Program Title and Objectives. The one thing that did need discovering:
the file isn't UTF-8 (a raw UnicodeDecodeError on a real row led to
checking) - it's cp1252, a common Windows/Excel CSV export encoding.

Reuses the same chunk shape as ingest_glossary.py/ingest_naics.py/
ingest_psc.py (id/source/page_start/page_end/paragraph_index/text/term/
slug/related_slugs) so vector_index.py/bm25_index.py need no changes -
term=title, slug=program number.

Usage:
  python -m backend.app.retrieval.pipeline.ingest_cfda
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

SOURCE_NAME = "CFDA Programs"

CFDA_FILE = "data/raw/AssistanceListings_DataGov_PUBLIC_CURRENT.csv"
CFDA_FILE_ENCODING = "cp1252"


@dataclass
class CFDAChunk:
    id: str
    source: str
    page_start: int
    page_end: int
    paragraph_index: int
    text: str
    term: str
    slug: str
    related_slugs: list[str]


def load_rows(path: str) -> list[dict]:
    with open(path, encoding=CFDA_FILE_ENCODING, newline="") as fh:
        return list(csv.DictReader(fh))


def build_chunk(row: dict) -> CFDAChunk:
    number = row["Program Number"].strip()
    title = row["Program Title"].strip()

    text_parts = [f"{number} - {title}"]
    popular_name = row["Popular Name (020)"].strip()
    if popular_name:
        text_parts.append(f"Also known as: {popular_name}")
    objectives = row["Objectives (050)"].strip()
    if objectives:
        text_parts.append(objectives)
    uses = row["Uses and Use Restrictions (070)"].strip()
    if uses:
        text_parts.append(f"Uses: {uses}")
    assistance_type = row["Types of Assistance (060)"].strip()
    if assistance_type:
        text_parts.append(f"Type of assistance: {assistance_type}")

    return CFDAChunk(
        id=f"CFDA_{number}",
        source=SOURCE_NAME,
        page_start=0,
        page_end=0,
        paragraph_index=0,
        text="\n\n".join(text_parts),
        term=title,
        slug=number,
        related_slugs=[],
    )


def ingest_cfda_to_chunks(out_path: Path) -> list[CFDAChunk]:
    rows = load_rows(CFDA_FILE)
    chunks = [build_chunk(row) for row in rows]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return chunks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/chunks/cfda_chunks.jsonl")
    args = parser.parse_args()

    chunks = ingest_cfda_to_chunks(Path(args.out))
    print(f"Wrote {len(chunks)} chunks to {args.out}")


if __name__ == "__main__":
    main()

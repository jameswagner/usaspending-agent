"""PDF ingestion and chunking utilities for the Analyst's Guide.

This module extracts text from a provided PDF, performs simple cleaning to
remove common headers/footers and page artifacts, and chunks text at a
paragraph/sentence-granularity into JSONL records with metadata.

Usage:
  python -m backend.app.retrieval.ingest --pdf data/raw/analysts_guide.pdf

Note: The actual PDF isn't included here — upload it to `data/raw/` and
then run the script. This scaffold focuses on deterministic, inspectable
chunking (no embeddings yet).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import pdfplumber


@dataclass
class Chunk:
    id: str
    source: str
    page_start: int
    page_end: int
    paragraph_index: int
    text: str


def extract_pages(pdf_path: Path) -> List[str]:
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for p in pdf.pages:
            text = p.extract_text() or ""
            pages.append(text)
    return pages


def detect_repeated_lines(pages: List[str], min_count: int = 3, max_line_len: int = 200) -> set:
    """Find short lines repeated across pages (likely headers/footers).

    Returns a set of lines to remove from page text.
    """
    lines = []
    for page in pages:
        for ln in page.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if len(ln) > max_line_len:
                continue
            # ignore numeric-only page numbers
            if re.fullmatch(r"\d+", ln):
                continue
            lines.append(ln)

    counts = Counter(lines)
    repeated = {ln for ln, c in counts.items() if c >= min_count}
    return repeated


def clean_page_text(page_text: str, repeated_lines: set) -> str:
    lines = []
    for ln in page_text.splitlines():
        ln_stripped = ln.strip()
        if not ln_stripped:
            lines.append("")
            continue
        # remove simple page markers and very short boilerplate
        if re.fullmatch(r"Page\s*\d+", ln_stripped, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", ln_stripped):
            continue
        if ln_stripped in repeated_lines:
            continue
        # drop lines that are just sequences of hyphens/asterisks
        if re.fullmatch(r"[-*_]{3,}", ln_stripped):
            continue
        lines.append(ln.rstrip())
    # join preserving paragraph breaks
    cleaned = "\n".join(lines)
    # collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def paragraph_split(text: str) -> List[str]:
    # split on two-or-more newlines or other common paragraph markers
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras


def sentence_split(paragraph: str) -> List[str]:
    # lightweight sentence split; good-enough for chunk boundaries
    sents = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in sents if s.strip()]


def chunk_paragraph(paragraph: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    # Build chunks by sentences until max_chars (approx). Include small overlap.
    sents = sentence_split(paragraph)
    chunks = []
    cur = []
    cur_len = 0
    for sent in sents:
        if cur_len + len(sent) + 1 <= max_chars or not cur:
            cur.append(sent)
            cur_len += len(sent) + 1
        else:
            chunks.append(" ".join(cur))
            # start next chunk with overlap sentences
            if overlap > 0:
                # include as many sentences from the tail as fit in overlap
                tail = []
                tail_len = 0
                for s in reversed(cur):
                    if tail_len + len(s) + 1 > overlap:
                        break
                    tail.insert(0, s)
                    tail_len += len(s) + 1
                cur = tail.copy()
                cur_len = tail_len
            else:
                cur = []
                cur_len = 0
            cur.append(sent)
            cur_len += len(sent) + 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def ingest_pdf_to_chunks(pdf_path: Path, out_path: Path, source_name: str = "Analyst's Guide") -> List[Chunk]:
    pages = extract_pages(pdf_path)
    repeated = detect_repeated_lines(pages)

    all_chunks: List[Chunk] = []
    chunk_id = 0
    for i, page_text in enumerate(pages, start=1):
        cleaned = clean_page_text(page_text, repeated)
        paras = paragraph_split(cleaned)
        for p_idx, para in enumerate(paras):
            para_chunks = chunk_paragraph(para)
            for sub_idx, ch_text in enumerate(para_chunks):
                chunk_id += 1
                c = Chunk(
                    id=f"{source_name.replace(' ', '_')}_p{chunk_id}",
                    source=source_name,
                    page_start=i,
                    page_end=i,
                    paragraph_index=p_idx,
                    text=ch_text,
                )
                all_chunks.append(c)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for c in all_chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    return all_chunks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, help="Path to analysts_guide.pdf")
    parser.add_argument("--out", default="data/chunks/analysts_guide_chunks.jsonl")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    out_path = Path(args.out)

    if not pdf_path.exists():
        print(f"PDF not found: {pdf_path}. Place the Analyst's Guide at this path and retry.")
        return

    chunks = ingest_pdf_to_chunks(pdf_path, out_path)
    print(f"Wrote {len(chunks)} chunks to {out_path}")


if __name__ == "__main__":
    main()

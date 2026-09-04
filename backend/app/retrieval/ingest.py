"""PDF ingestion and chunking utilities for the Analyst's Guide.

This module extracts text from a provided PDF, performs simple cleaning to
remove common headers/footers and page artifacts, and chunks text into
question/answer units (the guide is written as a Q&A doc, each question
wrapped in curly quotes and ending in '?') into JSONL records with metadata.

Usage:
  python -m backend.app.retrieval.ingest --pdf data/raw/analysts_guide.pdf
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf


@dataclass
class Chunk:
    id: str
    source: str
    page_start: int
    page_end: int
    paragraph_index: int
    text: str


# Curly-quoted questions used throughout the guide, e.g. 'What is a prime award?'.
# We match on the *start* of a question (opening quote + a question word) rather
# than pairing opening/closing quotes: the source PDF sometimes mis-renders a
# closing quote using the opening-quote glyph, which breaks quote-pair matching
# but leaves the start marker intact.
QUESTION_START_RE = re.compile(
    r"[‘']\s*(?=(?:What|How|Which|When|Where|Why|Who|Can|Could|Is|Are|Does|Do|Did|"
    r"Should|Would|Will|May|Must|I)\b)"
)


def extract_pages(pdf_path: Path) -> list[str]:
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text())
    return pages


def detect_repeated_lines(pages: list[str], min_count: int = 3, max_line_len: int = 200) -> set:
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
        # drop "Page N" / "PAGE N OF M" style footers regardless of the
        # page number, since the number makes each line unique and defeats
        # exact-repeat detection
        if re.fullmatch(r"page\s*\d+", ln_stripped, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"page\s*\d+\s*of\s*\d+", ln_stripped, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"\d+", ln_stripped):
            continue
        if ln_stripped in repeated_lines:
            continue
        if re.fullmatch(r"[-*_]{3,}", ln_stripped):
            continue
        lines.append(ln.rstrip())
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def paragraph_split(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    return paras


def question_split(text: str) -> list[str]:
    """Split cleaned page text into Q&A units at each curly-quoted question.

    Each unit starts at a question marker (e.g. 'What is a prime award?')
    and runs up to (but not including) the next question marker, so a
    question and its answer stay together in one chunk. Any text before the
    first question (section headers) is kept as its own leading unit.
    """
    matches = list(QUESTION_START_RE.finditer(text))
    if not matches:
        return [text.strip()] if text.strip() else []

    units = []
    lead = text[: matches[0].start()].strip()
    if lead:
        units.append(lead)

    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        unit = text[m.start():end].strip()
        if unit:
            units.append(unit)

    return units


def sentence_split(paragraph: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', paragraph)
    return [s.strip() for s in sents if s.strip()]


def chunk_unit(unit: str, max_chars: int = 1000, overlap: int = 200) -> list[str]:
    """Split a Q&A (or leading) unit into char-budgeted sub-chunks.

    Units under max_chars pass through unchanged (the common case, since
    most Q&A pairs in the guide are short). Longer units fall back to
    sentence-level packing with overlap, same as before.
    """
    if len(unit) <= max_chars:
        return [unit]

    sents = sentence_split(unit)
    chunks = []
    cur = []
    cur_len = 0
    for sent in sents:
        if cur_len + len(sent) + 1 <= max_chars or not cur:
            cur.append(sent)
            cur_len += len(sent) + 1
        else:
            chunks.append(" ".join(cur))
            if overlap > 0:
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


def ingest_pdf_to_chunks(pdf_path: Path, out_path: Path, source_name: str = "Analyst's Guide") -> list[Chunk]:
    pages = extract_pages(pdf_path)
    repeated = detect_repeated_lines(pages)

    all_chunks: list[Chunk] = []
    chunk_id = 0
    for i, page_text in enumerate(pages, start=1):
        cleaned = clean_page_text(page_text, repeated)
        units = question_split(cleaned)
        for u_idx, unit in enumerate(units):
            for sub_text in chunk_unit(unit):
                chunk_id += 1
                c = Chunk(
                    id=f"{source_name.replace(' ', '_')}_p{chunk_id}",
                    source=source_name,
                    page_start=i,
                    page_end=i,
                    paragraph_index=u_idx,
                    text=sub_text,
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
    parser.add_argument("--pdf", required=True, help="Path to analyst-guide.pdf")
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

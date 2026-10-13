"""Opens a finished download's ZIP and reports what is actually in it, since the job status alone only says it finished."""
from __future__ import annotations

import csv
import io
import logging
import zipfile
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger(__name__)

MAX_ZIP_BYTES = 25_000_000
MAX_COUNTED_ROWS = 1_000_000
SAMPLE_ROWS = 2
_FETCH_TIMEOUT_SECONDS = 30


@dataclass
class FileSummary:
    name: str
    rows: int
    columns: list[str]
    sample: list[list[str]]
    truncated: bool = False


@dataclass
class DownloadInspection:
    files: list[FileSummary] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    skipped: str | None = None

    def render(self) -> str:
        if self.skipped:
            return f"File check skipped: {self.skipped}."
        lines = []
        for f in self.files:
            count = f"{f.rows:,}{'+' if f.truncated else ''}"
            if f.rows == 0:
                lines.append(f"- {f.name}: 0 rows (empty)")
                continue
            sample = "; ".join(", ".join(cell[:40] for cell in row) for row in f.sample)
            lines.append(f"- {f.name}: {count} rows; columns: {', '.join(f.columns)}; first rows: {sample}")
        text = "File check (what the ZIP actually contains):\n" + "\n".join(lines)
        if self.problems:
            text += "\nProblems: " + " ".join(self.problems)
        return text


def _fetch_zip(url: str, max_bytes: int = MAX_ZIP_BYTES) -> bytes | str:
    """The ZIP's bytes, or a short reason the file was not fetched."""
    try:
        with httpx.stream("GET", url, timeout=_FETCH_TIMEOUT_SECONDS, follow_redirects=True) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > max_bytes:
                return f"the ZIP is {declared / 1e6:.0f} MB, over the {max_bytes / 1e6:.0f} MB inspection limit"
            buffer = io.BytesIO()
            for chunk in response.iter_bytes():
                buffer.write(chunk)
                if buffer.tell() > max_bytes:
                    return f"the ZIP is over the {max_bytes / 1e6:.0f} MB inspection limit"
            return buffer.getvalue()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Could not fetch download %s for inspection: %s", url, exc)
        return "the file could not be fetched for inspection"


def _summarize_csv(name: str, raw) -> FileSummary:
    reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
    header = next(reader, None)
    if header is None:
        return FileSummary(name=name, rows=0, columns=[], sample=[])
    rows, sample, truncated = 0, [], False
    for row in reader:
        rows += 1
        if len(sample) < SAMPLE_ROWS:
            sample.append(row)
        if rows >= MAX_COUNTED_ROWS:
            truncated = True
            break
    return FileSummary(name=name, rows=rows, columns=header, sample=sample, truncated=truncated)


def inspect_download(
    file_url: str, expected_columns: list[str] | None = None, expected_rows: int | None = None
) -> DownloadInspection:
    """Never raises: a file that cannot be inspected is reported as skipped, not as a failed download."""
    fetched = _fetch_zip(file_url)
    if isinstance(fetched, str):
        return DownloadInspection(skipped=fetched)
    try:
        with zipfile.ZipFile(io.BytesIO(fetched)) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
            files = [_summarize_csv(n, archive.open(n)) for n in names]
    except (zipfile.BadZipFile, UnicodeDecodeError, csv.Error) as exc:
        logger.warning("Could not read download %s: %s", file_url, exc)
        return DownloadInspection(skipped="the ZIP could not be read")

    inspection = DownloadInspection(files=files)
    if not files:
        inspection.problems.append("The ZIP contains no CSV files.")
        return inspection

    non_empty = [f for f in files if f.rows > 0]
    for f in files:
        if f.rows == 0 and non_empty:
            inspection.problems.append(
                f"{f.name} is empty even though other files have rows; that part of the request returned nothing."
            )
    if not non_empty:
        inspection.problems.append("Every file in the ZIP is empty, so this download returned no records.")
    else:
        present = {column for f in non_empty for column in f.columns}
        missing = [c for c in (expected_columns or []) if c not in present]
        if missing:
            inspection.problems.append(f"Requested columns missing from the files: {', '.join(missing)}.")
    total = sum(f.rows for f in files)
    if expected_rows is not None and expected_rows > 0 and total == 0:
        inspection.problems.append(f"The job reported {expected_rows} rows but the files contain none.")
    return inspection

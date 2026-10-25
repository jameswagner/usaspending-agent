"""Recognizing breakdown rows that are not real entities (MULTIPLE RECIPIENTS,
REDACTED DUE TO PII, ...) so they are reported on their own line instead of
ranking first as if they were a recipient, district, office or code.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

T = TypeVar("T")

_EXACT_NAMES = {
    "multiple recipients": "an aggregate bucket of many recipients, not one recipient",
    "redacted due to pii": "individual recipients whose names are redacted for privacy, not one recipient",
    "aggregate financial assistance": "a catch-all office for aggregated financial assistance, not a real awarding office",
    "not specified": "records with no value reported, not a real category",
    "not designated": "records with no disaster emergency fund code designation, not a real code",
}
_SUBSTRINGS = {
    "multiple districts": "awards spanning several congressional districts, not a real district",
}
_UNNAMED = "an unnamed row with no name or code, not an identifiable entity"


def placeholder_reason(name: str | None, code: str | None = None) -> str | None:
    if not (name or "").strip() and not (code or "").strip():
        return _UNNAMED
    for value in (name, code):
        text = (value or "").strip().lower()
        if text in _EXACT_NAMES:
            return _EXACT_NAMES[text]
        for fragment, reason in _SUBSTRINGS.items():
            if fragment in text:
                return reason
    return None


def format_with_placeholders(
    rows: Iterable[T],
    identify: Callable[[T], tuple[str | None, str | None]],
    format_row: Callable[[T], str],
) -> str:
    """Real rows first in their given order, then placeholders on their own
    lines with a plain explanation that they are excluded from any ranking."""
    real: list[str] = []
    placeholders: list[str] = []
    for row in rows:
        reason = placeholder_reason(*identify(row))
        if reason:
            placeholders.append(f"{format_row(row)} - {reason}")
        else:
            real.append(format_row(row))
    if not placeholders:
        return "\n".join(real)
    header = "Placeholder rows (not real entities; do not rank or present these as the top result):"
    return "\n".join([*real, header, *placeholders])


def split_placeholders(
    rows: Sequence[T], identify: Callable[[T], tuple[str | None, str | None]]
) -> tuple[list[T], list[T]]:
    real = [r for r in rows if placeholder_reason(*identify(r)) is None]
    return real, [r for r in rows if placeholder_reason(*identify(r)) is not None]

"""Local copy of upstream's api_contracts/ markdown, and a one-line-per-doc
catalog of it for the raw-API agent's system prompt.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

UPSTREAM_REPO = "https://github.com/fedspendingtransparency/usaspending-api.git"
CONTRACTS_SUBDIR = "usaspending_api/api_contracts"
CONTRACTS_DIR = Path(os.environ.get("API_CONTRACTS_DIR", "./data/api_contracts"))

# Bulk-job, write-ish, and not-yet-live endpoints the agent must never call.
DENIED_PATH_PREFIXES = (
    "/api/v2/download/",
    "/api/v2/bulk_download/",
    "/api/v2/llm/",
    "/api/v1/references/filter/",
)

_HEADING_PATH = re.compile(r"\[(?:(GET|POST)\s+)?(/api/[^\]\s]+)\]")
_METHOD_HEADING = re.compile(r"^##+\s+(GET|POST)\b", re.MULTILINE)


@dataclass(frozen=True)
class Endpoint:
    method: str
    path_template: str


@dataclass(frozen=True)
class ContractDoc:
    doc_id: str
    title: str
    summary: str
    endpoints: tuple[Endpoint, ...]
    unavailable: bool


def sync_contracts(dest: Path = CONTRACTS_DIR) -> str:
    """Sparse-clones just api_contracts/ and returns the upstream commit SHA."""
    checkout = dest.parent / f".{dest.name}_checkout"
    if checkout.exists():
        shutil.rmtree(checkout)
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse", UPSTREAM_REPO, str(checkout)],
        check=True,
    )
    subprocess.run(["git", "-C", str(checkout), "sparse-checkout", "set", CONTRACTS_SUBDIR], check=True)
    sha = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(checkout / CONTRACTS_SUBDIR), str(dest))
    shutil.rmtree(checkout)
    (dest / "UPSTREAM_COMMIT").write_text(sha + "\n")
    load_catalog.cache_clear()
    return sha


def _path_from_location(doc_id: str) -> str:
    return "/api/" + doc_id.removeprefix("contracts/").removesuffix(".md") + "/"


def _summary(body: str) -> str:
    for paragraph in re.split(r"\n\s*\n", body):
        text = " ".join(paragraph.split())
        if text and not text.startswith(("#", "+", "FORMAT:", "HOST:", "**Status")):
            return text[:160]
    return ""


def _parse_doc(doc_id: str, text: str) -> ContractDoc:
    title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    title_line = title_match.group(1) if title_match else doc_id
    title = _HEADING_PATH.sub("", title_line).strip()
    body = text[title_match.end():] if title_match else text

    endpoints: list[Endpoint] = []
    default_methods = _METHOD_HEADING.findall(text) or ["GET"]
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        for method, path in _HEADING_PATH.findall(line):
            path = re.sub(r"\{\?[^}]*\}", "", path)
            for m in [method] if method else default_methods:
                endpoints.append(Endpoint(m, path))
    if not endpoints:
        endpoints = [Endpoint(m, _path_from_location(doc_id)) for m in default_methods]

    unavailable = bool(re.search(r"NOT AVAILABLE|\[DEPRECATED\]", title_line, re.IGNORECASE))
    return ContractDoc(doc_id, title, _summary(body), tuple(dict.fromkeys(endpoints)), unavailable)


@lru_cache(maxsize=1)
def load_catalog(root: Path = CONTRACTS_DIR) -> dict[str, ContractDoc]:
    if not (root / "contracts").is_dir():
        raise FileNotFoundError(
            f"No API contracts at {root} - run: uv run python -m backend.app.raw_api_agent sync"
        )
    docs = {}
    for md in sorted((root / "contracts").rglob("*.md")):
        doc_id = md.relative_to(root).as_posix()
        docs[doc_id] = _parse_doc(doc_id, md.read_text(encoding="utf-8"))
    return docs


def upstream_commit(root: Path = CONTRACTS_DIR) -> str:
    marker = root / "UPSTREAM_COMMIT"
    return marker.read_text().strip() if marker.exists() else "unknown"


def read_doc(doc_id: str, root: Path = CONTRACTS_DIR) -> str | None:
    """Any catalog doc, plus the top-level reference docs like search_filters.md."""
    candidate = (root / doc_id).resolve()
    if root.resolve() not in candidate.parents or candidate.suffix != ".md" or not candidate.is_file():
        return None
    text = candidate.read_text(encoding="utf-8")
    return re.sub(r"\A(FORMAT:.*\n)?(HOST:.*\n)?\s*", "", text)


def _template_regex(template: str) -> re.Pattern:
    escaped = re.escape(template.rstrip("/"))
    return re.compile("^" + re.sub(r"\\\{[^}]*\\\}", r"[^/]+", escaped) + "/?$")


@lru_cache(maxsize=1)
def _allowlist() -> tuple[tuple[str, re.Pattern, str], ...]:
    return tuple(
        (ep.method, _template_regex(ep.path_template), doc.doc_id)
        for doc in load_catalog().values()
        if not doc.unavailable
        for ep in doc.endpoints
        if not ep.path_template.startswith(DENIED_PATH_PREFIXES)
    )


def match_endpoint(method: str, path: str) -> str | None:
    """The doc_id documenting this method+path, or None if it's not callable."""
    if path.startswith(DENIED_PATH_PREFIXES):
        return None
    for allowed_method, pattern, doc_id in _allowlist():
        if allowed_method == method and pattern.match(path):
            return doc_id
    return None


def catalog_text() -> str:
    lines = []
    for doc in load_catalog().values():
        callable_endpoints = [ep for ep in doc.endpoints if not ep.path_template.startswith(DENIED_PATH_PREFIXES)]
        if doc.unavailable or not callable_endpoints:
            continue
        calls = ", ".join(f"{ep.method} {ep.path_template}" for ep in callable_endpoints)
        lines.append(f"- {doc.doc_id} | {calls} | {doc.title}: {doc.summary}")
    return "\n".join(lines)

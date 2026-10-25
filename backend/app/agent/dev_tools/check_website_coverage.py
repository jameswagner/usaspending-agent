"""Endpoint coverage against usaspending.gov: which API endpoints the website
calls (per the website audit's inventory) that no backend code calls.

Static and free - no API calls. Both tool sets share the same client, so one
answer covers both:
    uv run python -m backend.app.agent.dev_tools.check_website_coverage
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from backend.app.agent.tools.spending.category import VALID_CATEGORIES

REPO_ROOT = Path(__file__).resolve().parents[4]
INVENTORY_PATH = REPO_ROOT / "docs/research/usaspending-website-audit-2026-10-17/inventory.json"
BACKEND_DIRS = [REPO_ROOT / "backend/app/usaspending", REPO_ROOT / "backend/app/agent"]
EXCLUDED_DIRS = {"dev_tools", "__pycache__"}

# Website endpoints that serve page mechanics, not a question someone would ask in chat. "*" is one path segment.
PLUMBING = {
    "/api/v2/references/filter/": "saves a shareable search link",
    "/api/v2/references/hash/": "opens a saved search link",
    "/api/v2/references/filter_tree/*/": "filter picker tree",
    "/api/v2/references/filter_tree/*/*/": "filter picker tree",
    "/api/v2/references/states/": "state dropdown",
    "/api/v2/references/submission_periods/": "period picker",
    "/api/v2/autocomplete/glossary/": "glossary search box (search_guide covers the glossary)",
    "/api/v2/awards/count/*/*/": "tab count badge",
    "/api/v2/idvs/count/*/*/": "tab count badge",
    "/api/v2/disaster/*/count/": "table row count",
    "/api/v2/search/spending_by_transaction_count/": "tab count badge",
}
AI_SEARCH = "/api/v2/llm/filter-search/"

# Website endpoints whose question the backend answers from bundled or ingested data instead of a live call.
LOCAL = {
    "/api/v2/budget_functions/list_budget_functions/": "resolve_budget_function's bundled list",
    "/api/v2/budget_functions/list_budget_subfunctions/": "resolve_budget_function's bundled list",
    "/api/v2/references/glossary/": "search_guide's ingested glossary",
    "/api/v2/autocomplete/cfda/": "resolve_cfda_program's local index",
    "/api/v2/autocomplete/naics/": "resolve_naics_code's local index",
    "/api/v2/references/naics/": "resolve_naics_code's local index",
    "/api/v2/references/naics/*/": "resolve_naics_code's local index",
    "/api/v2/autocomplete/recipient/": "search_recipients (/api/v2/recipient/)",
    "/api/v2/autocomplete/awarding_agency/": "agency names resolved against /api/v2/references/toptier_agencies/",
    "/api/v2/autocomplete/funding_agency/": "agency names resolved against /api/v2/references/toptier_agencies/",
}
STATUSES = ["covered", "local", "gap", "plumbing", "dead", "ai_search"]

_PATH_LITERAL = re.compile(r"""f?["'](/api/v[12]/[^"'\s]*)["']""")
_PLACEHOLDER = re.compile(r"^\{+[^{}]*\}+$")


def _segments(path: str) -> list[str]:
    return [s for s in path.split("?")[0].strip("/").split("/") if s]


def backend_endpoints() -> list[list[str]]:
    """Every API path literal in backend code, as segments; a {category} segment expands to each valid category."""
    templates: set[tuple[str, ...]] = set()
    for root in BACKEND_DIRS:
        for file in root.rglob("*.py"):
            if EXCLUDED_DIRS & set(file.relative_to(root).parts):
                continue
            for literal in _PATH_LITERAL.findall(file.read_text(encoding="utf-8")):
                segments = _segments(literal)
                if "{category}" in segments:
                    i = segments.index("{category}")
                    templates.update(tuple(segments[:i] + [c] + segments[i + 1:]) for c in VALID_CATEGORIES)
                else:
                    templates.add(tuple(segments))
    return [list(t) for t in sorted(templates)]


def _is_identifier(segment: str) -> bool:
    """An ID or code like 080, CONT_AWD_..., 2025 - not an endpoint name like funding_rollup."""
    return bool(_PLACEHOLDER.match(segment)) or any(c.isdigit() or c.isupper() for c in segment)


def _matches(path: list[str], pattern: list[str], wildcard, identifiers_only: bool = False) -> bool:
    if len(path) != len(pattern):
        return False
    return all(p == q or (wildcard(q) and (not identifiers_only or _is_identifier(p))) for p, q in zip(path, pattern))


def classify(task: dict, backend: list[list[str]]) -> tuple[str, str]:
    """(status, reason) for one inventory task, status being one of STATUSES."""
    path = task["request"]["path"]
    segments = _segments(path)
    if task.get("ui_dead_code"):
        return "dead", "the website never calls it"
    if path.split("?")[0] == AI_SEARCH:
        return "ai_search", "the website's own AI search"
    if any(_matches(segments, b, lambda s: bool(_PLACEHOLDER.match(s)), identifiers_only=True) for b in backend):
        return "covered", ""
    for status, table in (("local", LOCAL), ("plumbing", PLUMBING)):
        for pattern, reason in table.items():
            if _matches(segments, _segments(pattern), lambda s: s == "*"):
                return status, reason
    return "gap", ""


def load_tasks() -> list[dict]:
    return json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))["tasks"]


def report(tasks: list[dict], backend: list[list[str]]) -> str:
    rows = [(t, *classify(t, backend)) for t in tasks]
    by_section: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for task, status, _ in rows:
        by_section[task["section"]][status] += 1
    totals = {s: sum(1 for _, status, _ in rows if status == s) for s in STATUSES}
    reached = totals["covered"] + totals["local"]
    relevant = reached + totals["gap"]
    summary = (
        f"Website tasks: {len(rows)}. Chat-relevant: {relevant}, of which {reached} are reachable "
        f"({100 * reached / relevant:.0f}%: {totals['covered']} by a live call, {totals['local']} from local data) "
        f"and {totals['gap']} aren't."
    )
    lines = [
        summary,
        "",
        "| section | " + " | ".join(STATUSES) + " |",
        "|---" * (len(STATUSES) + 1) + "|",
    ]
    for section, counts in sorted(by_section.items(), key=lambda kv: -kv[1]["gap"]):
        lines.append(f"| {section} | " + " | ".join(str(counts[s] or "") for s in STATUSES) + " |")
    lines += ["", "## Gaps", ""]
    for task, status, _ in rows:
        if status == "gap":
            lines.append(f"- {task['id']} `{task['request']['path']}`: {task['question']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--status", choices=STATUSES, help="list only these tasks")
    args = parser.parse_args()
    tasks, backend = load_tasks(), backend_endpoints()
    if args.status:
        for task in tasks:
            status, reason = classify(task, backend)
            if status == args.status:
                print(f"{task['id']} {task['request']['path']}  {reason or task['question']}")
        return
    print(report(tasks, backend))


if __name__ == "__main__":
    main()

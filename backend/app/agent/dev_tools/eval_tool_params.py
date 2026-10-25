"""Tool-and-parameter eval over a recorded compare_raw_api run: did each arm call
the right tool, in the right mode, with the right arguments?

Offline and free - it scores the trajectories compare_raw_api already saved, so
the same run can be rescored as the labeled set changes. Calls from both tool
sets are first mapped onto one canonical vocabulary (the consolidated names plus
section/kind), so a single expectation covers both:
get_agency_budget(group_by="sub_component") and
agency_profile(section="budget_by_bureau") both become
"agency_profile.budget_by_bureau".

    uv run python -m backend.app.agent.dev_tools.eval_tool_params data/raw_api_eval/compare-<stamp>.jsonl

Entries with a 'reference' are also checked for the right answer, not just the
right call: --check-references replays each reference request against the live
API and looks for the extracted value in the arm's answer.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import requests

LABELED_SET_PATH = Path(__file__).parent / "tool_params_labeled_set.json"
PRESENT = "__present__"
ARM_LABELS = {"a": "current", "b": "raw-api", "c": "consolidated"}

_AGENCY_AWARD_SECTIONS = {"sub_agency": "awards_by_sub_agency", "award_category": "awards_by_category", "whole_agency": "award_activity"}
_RESOLVERS = {
    "resolve_naics_code": "naics", "resolve_psc_code": "psc", "resolve_cfda_program": "cfda",
    "resolve_county_fips": "county", "resolve_budget_function": "budget_function",
}
_AWARD_SECTIONS = {
    "get_award_details": "summary", "get_award_transaction_history": "transactions",
    "get_award_subawards": "subawards", "get_award_funding_breakdown": "funding",
}
_DEFAULT_SECTIONS = {"award_detail": "summary", "recipient_profile": "details"}


def canonicalize(tool: str, args: dict) -> tuple[str, dict]:
    """Maps a call from either tool set to (action, args) in the consolidated vocabulary."""
    args = {k: v for k, v in (args or {}).items() if v is not None}
    if tool == "lookup_agency":
        return "agency_profile.overview", {"agency_name": args.get("name")}
    if tool == "get_agency_budget":
        renamed = {**args, "fiscal_year": args.get("start_fiscal_year")}
        renamed.pop("start_fiscal_year", None)
        section = "budget_by_bureau" if renamed.pop("group_by", None) == "sub_component" else "budget"
        return f"agency_profile.{section}", renamed
    if tool == "get_agency_budget_by_subcomponent":
        return "agency_profile.budget_by_bureau", args
    if tool == "get_agency_award_breakdown":
        section = _AGENCY_AWARD_SECTIONS[args.pop("group_by", "sub_agency")]
        return f"agency_profile.{section}", args
    if tool == "list_top_agencies_by_budget":
        return "agency_profile.rank_all_agencies", args
    if tool in _RESOLVERS:
        return f"resolve_entity.{_RESOLVERS[tool]}", {"query": args.get("description")}
    if tool == "search_recipients":
        return "resolve_entity.recipient", {"query": args.get("keyword")}
    if tool in _AWARD_SECTIONS:
        return f"award_detail.{_AWARD_SECTIONS[tool]}", args
    if tool == "get_recipient_details":
        return "recipient_profile.details", args
    if tool == "get_recipient_children":
        return "recipient_profile.children", args
    if tool == "get_award_type_breakdown":
        return "query_spending", {**args, "group_by": "award_type"}
    if tool == "resolve_entity":
        return f"resolve_entity.{args.pop('kind', None)}", args
    if tool in ("agency_profile", "award_detail", "recipient_profile"):
        section = args.pop("section", None) or _DEFAULT_SECTIONS.get(tool)
        return f"{tool}.{section}", args
    return tool, args


def _normalize(value):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    return str(value).strip().lower()


def _value_matches(actual, expected, earlier: dict[str, list[str]] | None = None) -> bool:
    if expected == PRESENT:
        return actual is not None
    if isinstance(expected, dict) and "any_of" in expected:
        return any(_value_matches(actual, option, earlier) for option in expected["any_of"])
    if isinstance(expected, dict) and "from" in expected:
        outputs = (earlier or {}).get(expected["from"], [])
        values = actual if isinstance(actual, list) else [actual]
        return actual is not None and all(any(str(v) in output for output in outputs) for v in values)
    return actual is not None and _normalize(actual) == _normalize(expected)


def _call_satisfies(call: dict, rule: dict, earlier: dict[str, list[str]] | None = None) -> list[str]:
    """Returns why call fails rule's args/absent constraints; empty when it satisfies them."""
    problems = []
    for key, expected in rule.get("args", {}).items():
        if not _value_matches(call["args"].get(key), expected, earlier):
            problems.append(f"{key}={call['args'].get(key)!r}, wanted {expected!r}")
    problems += [f"{key}={call['args'][key]!r} should be unset" for key in rule.get("absent", []) if key in call["args"]]
    return problems


def _actions(step: dict) -> set[str]:
    action = step["action"]
    return set(action) if isinstance(action, list) else {action}


def score_trajectory(trajectory: list[dict], entry: dict) -> dict:
    calls = [
        {
            **dict(zip(("action", "args"), canonicalize(s["tool"], s.get("args") or {}))),
            "rejected": (s.get("output") or "").startswith(("This query failed", "This download failed")),
            "output": s.get("output") or "",
        }
        for s in trajectory
    ]
    earlier_at: list[dict[str, list[str]]] = []
    seen: dict[str, list[str]] = defaultdict(list)
    for c in calls:
        earlier_at.append({action: list(outputs) for action, outputs in seen.items()})
        seen[c["action"]].append(c["output"])
    problems: list[str] = []
    action_ok = True
    matched_positions = []
    for step in entry.get("expect", []):
        options = step.get("one_of", [step])
        candidates = [(i, option) for option in options for i, c in enumerate(calls) if c["action"] in _actions(option)]
        if not candidates:
            action_ok = False
            problems.append(f"no call to {'/'.join(sorted(set().union(*(_actions(o) for o in options))))}")
            continue
        exact = [i for i, option in candidates if not _call_satisfies(calls[i], option, earlier_at[i])]
        if exact:
            matched_positions.append(exact[0])
        else:
            index, option = candidates[0]
            problems.append(f"{calls[index]['action']}: " + "; ".join(_call_satisfies(calls[index], option, earlier_at[index])))
    if entry.get("ordered") and matched_positions != sorted(matched_positions):
        problems.append("expected calls out of order")
    for rule in entry.get("every_call", []):
        for i, call in enumerate(calls):
            if call["action"] in _actions(rule):
                problems += [f"{call['action']}: {p}" for p in _call_satisfies(call, rule, earlier_at[i])]
    for call in calls:
        if call["action"] in set(entry.get("forbid_actions", [])):
            problems.append(f"forbidden call to {call['action']}")
    rejected = [c for c in calls if c["rejected"]]
    if rejected and not entry.get("expect_error"):
        problems += [f"rejected: {c['action']}: {c['output'][:120]}" for c in rejected]
    return {
        "action_ok": action_ok,
        "params_ok": action_ok and not problems,
        "problems": problems,
        "calls": len(calls),
        "rejected": len(rejected),
    }


API_BASE = "https://api.usaspending.gov"
_UNITS = {"trillion": 1e12, "tn": 1e12, "t": 1e12, "billion": 1e9, "bn": 1e9, "b": 1e9,
          "million": 1e6, "m": 1e6, "thousand": 1e3, "k": 1e3}
_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(trillion|billion|million|thousand|tn|bn|[tbmk])?\b", re.IGNORECASE)


def replay(request: dict) -> dict:
    method = "POST" if request.get("body") is not None else "GET"
    response = requests.request(method, API_BASE + request["path"], params=request.get("query"), json=request.get("body"), timeout=120)
    response.raise_for_status()
    return response.json()


def extract(data, path: str):
    """Dotted path into a response; a '?key=value' or '?key!=value' component filters a list."""
    for part in path.split("."):
        if part.startswith("?"):
            key, negate, value = re.match(r"\?(\w+)(!?)=(.*)", part).groups()
            data = [item for item in data if (str(item.get(key)) == value) != bool(negate)]
        else:
            data = data[int(part)] if isinstance(data, list) else data[part]
    return data


def answer_amounts(answer: str) -> list[float]:
    amounts = []
    for number, unit in _AMOUNT.findall(answer):
        amounts.append(float(number.replace(",", "")) * _UNITS.get(unit.lower(), 1))
    return amounts


def _replayed_value(ref: dict, cache: dict[str, dict]):
    key = json.dumps(ref["request"], sort_keys=True)
    if key not in cache:
        cache[key] = replay(ref["request"])
    return extract(cache[key], ref["extract"])


def check_references(answer: str, entry: dict, cache: dict[str, dict]) -> dict:
    """Is each reference value, replayed live, stated in the answer?"""
    problems = []
    for ref in entry["reference"]:
        value = _replayed_value(ref, cache)
        label = ref.get("label", ref["extract"])
        kind = ref.get("kind", "amount")
        if kind == "text":
            accepted = [str(value)] + ref.get("aliases", [])
            if not any(a.lower() in answer.lower() for a in accepted):
                problems.append(f"{label}: expected {value!r} in the answer")
        elif kind == "percent":
            share = 100 * value / _replayed_value(ref["of"], cache)
            stated = [float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", answer)]
            if not any(abs(p - share) <= ref.get("tolerance", 0.5) for p in stated):
                problems.append(f"{label}: expected about {share:.1f}%")
        else:
            tolerance = ref.get("tolerance", 0.01)
            if not any(abs(a - value) <= tolerance * abs(value) for a in answer_amounts(answer)):
                problems.append(f"{label}: expected about {value:,.0f}")
    problems += [f"answer says {text!r}" for text in entry.get("answer_excludes", []) if text.lower() in answer.lower()]
    return {"truth_ok": not problems, "truth_problems": problems}


def load_labeled_set() -> dict[str, dict]:
    return {e["question"]: e for e in load_entries()}


def load_entries() -> list[dict]:
    with LABELED_SET_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["entries"]


def score_run(rows: list[dict], labeled: dict[str, dict], arms: list[str], check_refs: bool = False) -> list[dict]:
    scored = []
    cache: dict[str, dict] = {}
    for row in rows:
        entry = labeled.get(row["question"])
        if entry is None:
            continue
        result = {"question": row["question"], "category": row["category"]}
        for arm in arms:
            if "trajectory" in row.get(arm, {}):
                result[arm] = score_trajectory(row[arm]["trajectory"], entry)
                if check_refs and entry.get("reference"):
                    result[arm].update(check_references(row[arm].get("answer") or "", entry, cache))
        scored.append(result)
    return scored


def summarize(scored: list[dict], arms: list[str]) -> str:
    lines = ["| metric | " + " | ".join(ARM_LABELS[a] for a in arms) + " |", "|---" * (len(arms) + 1) + "|"]
    for label, key in (("right tool + mode", "action_ok"), ("right tool + mode + params", "params_ok"), ("right answer (reference)", "truth_ok")):
        cells = []
        for arm in arms:
            results = [s[arm][key] for s in scored if key in s.get(arm, {})]
            cells.append(f"{sum(results)}/{len(results)}")
        if key != "truth_ok" or any(c != "0/0" for c in cells):
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
    totals = defaultdict(lambda: [0, 0])
    for s in scored:
        for arm in arms:
            if arm in s:
                totals[arm][0] += s[arm]["calls"]
                totals[arm][1] += s[arm]["rejected"]
    lines.append("| calls (rejected) | " + " | ".join(f"{totals[a][0]} ({totals[a][1]})" for a in arms) + " |")
    lines += ["", "| category | " + " | ".join(ARM_LABELS[a] for a in arms) + " | problems |", "|---" * (len(arms) + 2) + "|"]
    for s in scored:
        marks = ["ok" if s.get(a, {}).get("params_ok") else ("mode ok" if s.get(a, {}).get("action_ok") else "wrong") for a in arms]
        marks = [m + ({True: ", answer ok", False: ", answer wrong"}.get(s.get(a, {}).get("truth_ok"), "")) for m, a in zip(marks, arms)]
        notes = "; ".join(
            f"{ARM_LABELS[a]}: {p}" for a in arms for p in s.get(a, {}).get("problems", []) + s.get(a, {}).get("truth_problems", [])
        )
        lines.append(f"| {s['category']} | " + " | ".join(marks) + f" | {notes.replace('|', '/')} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path, help="a compare_raw_api compare-<stamp>.jsonl file")
    parser.add_argument("--arms", default="ac", help="arms to score, e.g. 'ac'")
    parser.add_argument("--check-references", action="store_true", help="replay reference requests live and check answers")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.run.read_text(encoding="utf-8").splitlines() if line.strip()]
    labeled = load_labeled_set()
    scored = score_run(rows, labeled, list(args.arms), check_refs=args.check_references)
    missing = len(rows) - len(scored)
    print(summarize(scored, list(args.arms)))
    if missing:
        print(f"\n{missing} question(s) in the run have no labeled entry and were skipped.")


if __name__ == "__main__":
    main()

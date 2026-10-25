"""Render README.md from inventory.json and the replay evidence."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAME_KEYS = ['Recipient Name', 'Sub-Awardee Name', 'name', 'agency_name', 'recipient_name', 'title', 'description', 'display_name', 'Award ID', 'code', 'term', 'tas']
AMOUNT_KEYS = ['Award Amount', 'Loan Value', 'Sub-Award Amount', 'Transaction Amount', 'aggregated_amount', 'amount', 'obligation', 'transaction_obligation',
               'subaward_obligation', 'total_obligations', 'obligated_amount', 'budgetary_resources', 'total_budgetary_resources', 'face_value_of_loan',
               'federal_action_obligation', 'aggregate', 'count', 'total']
BASIS = {'browser_observed': 'browser', 'source_traced': 'source', 'source_dead_code': 'dead code'}


MONEY_KEY = re.compile(r'amount|obligat|outlay|budget|resource|value|aggregate|loan|total$', re.IGNORECASE)


def money(value, key=''):
    if key and (not MONEY_KEY.search(key) or 'count' in key.lower()):
        return f'{value:,}' if isinstance(value, int) and not re.search(r'year|id$|code', key, re.IGNORECASE) else str(value)
    if isinstance(value, (int, float)) and abs(value) >= 1000:
        for size, suffix in [(1e12, 'T'), (1e9, 'B'), (1e6, 'M'), (1e3, 'K')]:
            if abs(value) >= size:
                return f'${value / size:.2f}{suffix}'
    return str(value)


def row_summary(row):
    if not isinstance(row, dict):
        return str(row)[:80]
    name = next((str(row[k]) for k in NAME_KEYS if row.get(k) not in (None, '')), None)
    amount = next(((k, row[k]) for k in AMOUNT_KEYS if isinstance(row.get(k), (int, float))), None)
    parts = [name[:60] if name else None, f'{amount[0]} {money(amount[1], amount[0])}' if amount else None]
    return ', '.join(p for p in parts if p) or ', '.join(f'{k}={v}' for k, v in list(row.items())[:2])[:100]


def summarize(response):
    if isinstance(response, list):
        return f'{len(response)} rows; first: {row_summary(response[0])}' if response else '0 rows'
    for key in ['results', 'children', 'codes', 'monthly_files', 'terms']:
        value = response.get(key)
        if isinstance(value, list):
            total = (response.get('page_metadata') or {}).get('total') or response.get('total')
            head = f'{len(value)} rows' + (f' of {total:,}' if isinstance(total, int) else '')
            return f'{head}; top: {row_summary(value[0])}' if value else f'{head} (empty)'
    if 'results' in response and isinstance(response['results'], dict):
        return ', '.join(f'{k}={money(v, k)}' for k, v in response['results'].items())[:160]
    scalars = [(k, v) for k, v in response.items() if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
    return '; '.join(f'{k}={money(v, k)}' for k, v in scalars[:4])[:200] or ', '.join(list(response)[:6])


def main():
    inventory = json.loads((ROOT / 'inventory.json').read_text())
    meta, tasks = inventory['metadata'], inventory['tasks']
    evidence = {}
    for task in tasks:
        path = ROOT / 'evidence' / f'{task["id"]}.json'
        if path.exists():
            evidence[task['id']] = json.loads(path.read_text())
    ok = sum(1 for e in evidence.values() if e.get('status') == 'live_ok')
    runnable = sum(t['execute'] for t in tasks)
    counts = {b: sum(t['evidence_basis'] == b for t in tasks) for b in BASIS}

    out = [
        '# USAspending website audit (2026-10-17)',
        '',
        f'A list of {len(tasks)} things a person can do on usaspending.gov (search awards, open a profile, explore budgets, download data), each tied to the API request the website sends for it. The point is to measure the chat agent against the website: what it can reach, where its numbers could differ, and what to build next.',
        '',
        '## How it was checked',
        '',
        f'1. **Website code.** Every page traced to the helper that calls the API, at fedspendingtransparency/usaspending-website `{meta["website_commit"][:10]}` (API contracts at `{meta["api_commit"][:10]}`).',
        f'2. **Live replay.** {ok}/{runnable} runnable requests returned HTTP 2xx; each request and response is saved under `evidence/`. The other {len(tasks) - runnable} are export jobs or the website\'s key-gated AI search, traced but not run.',
        f'3. **Browser capture.** Playwright loaded 21 live pages and ran 2 click-through flows, logging every API call. {counts["browser_observed"]} tasks were seen this way; {counts["source_traced"]} need typing, a click, or a submit and are traced from code only; {counts["source_dead_code"]} use a helper the website never calls.',
        '',
        f'Baseline: {meta["baseline"]}',
        '',
        '## What matters for this repo',
        '',
        '1. **District rankings lead with a catch-all bucket.** For California FY2025 award obligations by congressional district, the top row is `CA-MULTIPLE DISTRICTS` at $203.3B; the largest real district, CA-36, has $7.8B (Q102). Nothing in the backend separates it out.',
        '2. **One search page shows two measures.** The Advanced Search results table uses `spending_level: "awards"` (lifetime award value); the time, category, and map charts beside it use `"transactions"` (obligations dated in the period). `spending_by_award` rejects `"transactions"`. NASA\'s top FY2025 contract shows $22.65B lifetime (Q001); all NASA FY2025 contract obligations total $15.78B (Q028). Answers need to name their measure (#230).',
        '3. **IDV amounts depend on vehicle type.** NASA\'s largest IDIQs (80JSC019C0012, NNJ16GU21B, NNJ16GX08B) report zero child awards; the money is on the vehicle. SEWP V GWAC NNG15SE16B shows a $0 award amount with 43 child orders carrying the obligations (Q006, Q073).',
        '4. **Downloads take a different agency ID.** `list_monthly_files`, `bulk_download/awards`, and `download/accounts` take `toptier_agency_id` (NASA 72) from `bulk_download/list_agencies`, not the `agency_id` (NASA 862) from `/agency/080/` (Q136, Q140).',
        '5. **Cheap download helpers.** `/download/count/` sizes an export in under a second (21,273 transactions for NASA FY2025 contracts, limit 500,000; Q141). `/bulk_download/list_monthly_files/` lists pre-built per-agency, per-year archives (Q140).',
        '6. **Website links need the website\'s filter format.** `/references/filter/` stores any JSON, but the site only opens hashes holding its own Redux state (`version: "2020-06-01"`) and sends anything else to `/hash-error`. `browser_capture_interactive.json` has a real payload to copy for "see this on USAspending.gov" links.',
        '7. **Federal account pages use v1.** `/api/v1/tas/balances/quarters/total/` (16 calls per page load) and `/api/v1/tas/categories/total/` drive the account charts and have no `api_contracts` entry. The v2 account endpoints all worked live (Q105, Q107, Q108).',
        '8. **Coverage.** 83 tasks hit an endpoint the backend already calls; 23 are website plumbing; 6 are dead on the site; 1 is the website\'s AI search; 50 are gaps. The chat-relevant gaps are disaster breakdowns (#110), federal/Treasury accounts (#303, #184), IDV child awards, award funding accounts, and state profiles (#119).',
        '',
        'Smaller notes: state profiles mix windows (top-5 tables use a trailing 12 months, header totals the last full FY, and the % of total divides one by the other; the API\'s `year=latest` for a state is the in-progress FY); the state "Awards" top-5 is always contracts ranked by lifetime award amount, labeled Obligations; the state DEFC chart uses `spending_level: "award_financial"`; the site never calls `spending_by_transaction_grouped`, `autocomplete/psc`, `autocomplete/glossary`, `awards/last_updated`, `agency/{code}/sub_agency/count`, or `agency/{code}/federal_account/count`.',
        '',
        '## Upstream quirks (don\'t affect this repo)',
        '',
        '- **Quoted award IDs return 503.** The contract says a double-quoted ID in `award_ids` forces an exact match; live, every quoted request returned an empty HTTP 503. The backend and the website both send IDs unquoted (Q024 notes).',
        '- **A 422 message mislabels two award-type groups.** Award search accepts types from one group per request. The error listing the groups names 06/10 `other_financial_assistance` and 09/11/-1 `direct_payments`, because `raise_if_award_types_not_valid_subset` in `usaspending_api/common/helpers/api_helper.py` passes those two mappings in the wrong order. The check itself is correct (Q021).',
        '',
        '## Files',
        '',
        '- `inventory.json`: every task with request, expected result, website source location, contract link, UI call sites, and `evidence_basis`.',
        '- `evidence/Q###.json`: the exact live request and response for each executed task.',
        '- `browser_capture.json`, `browser_capture_interactive.json`: API calls observed from 21 live pages and two click-through flows.',
        '- `build_inventory.py`: regenerates `inventory.json` (expects upstream checkouts under `/tmp/usaspending-website-audit-20261017/`).',
        '- `replay.py --phase N`: replays tasks; phases 2 and 3 use IDs extracted into `context.json` by earlier phases.',
        '- `render_report.py`: regenerates this file.',
        '',
    ]
    section = None
    for task in tasks:
        if task['section'] != section:
            section = task['section']
            out += ['', f'## {section}', '', '| ID | Question | Call | Basis | Live result |', '|---|---|---|---|---|']
        e = evidence.get(task['id'])
        if not task['execute']:
            result = 'not run: ' + (task['notes'] or 'export job')
        elif not e:
            result = 'no evidence'
        elif e.get('status') != 'live_ok':
            result = f'{e.get("status")} {e.get("http_status", "")}'
        else:
            result = summarize(e['response'])
        call = f'`{task["request"]["method"]} {task["request"]["path"].removeprefix("/api")}`'
        cells = [task['id'], task['question'], call, BASIS[task['evidence_basis']], result]
        out.append('| ' + ' | '.join(str(c).replace('|', '\\|').replace('\n', ' ') for c in cells) + ' |')
    (ROOT / 'README.md').write_text('\n'.join(out) + '\n')
    print(f'Wrote README.md: {len(tasks)} tasks, {ok}/{runnable} live OK')


if __name__ == '__main__':
    main()

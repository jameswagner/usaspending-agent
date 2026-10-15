# resolve-agency-code
use_when: any endpoint path containing {toptier_code}, or any filter that needs an agency's code

## Steps
1. Leave `{toptier_code}` in the path and pass the agency's name or abbreviation as `agency`
   (e.g. `agency="NSF"`). The tool fills in the code, including for sub-agencies like NIH.
2. For search filters (a JSON body, no path), use `filters.agencies[].name` with the agency's
   name instead; no code is needed.

## Pitfalls
- Never guess or type a code yourself, and don't probe `/api/v2/references/agency/{n}/` with guessed ids.
- `agency_id` (e.g. 1525) is a different number from `toptier_code`. Never put it in a {toptier_code} path.
- A 4-digit code like "4900" is a sub-tier code, not a toptier code.
- Sub-agencies (NIH, FAA, ...) have no toptier_code of their own; `agency` resolves them to their parent's.
  On search endpoints use `filters.agencies[].tier = "subtier"` to target the sub-agency itself.

verified: contracts@03b9e255, live API 2026-10-14

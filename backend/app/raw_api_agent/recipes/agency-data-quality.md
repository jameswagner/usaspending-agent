# agency-data-quality
use_when: how reliable, complete, accurate, or timely an agency's USAspending data is; unlinked awards; late or missing submissions; reporting discrepancies
not_when: how much an agency spent or obligated (that's spending data, not reporting metadata)

These questions ARE in scope - USAspending publishes this under "About the Data".

## Fiscal periods
`fiscal_period` runs 2-12 (2 = November ... 12 = September). Quarter Q -> period Q*3, so a
full-year / FY-end question uses period 12.

## Which endpoint
- Every agency at once, ranked: `GET /api/v2/reporting/agencies/overview/?fiscal_year=&fiscal_period=`
  (limit max 100; ~111 agencies, so fetch page 2 too, or use `sort` + `order`).
- One agency over time: `GET /api/v2/reporting/agencies/{toptier_code}/overview/` (see resolve-agency-code).
- Unlinked award counts with their denominator:
  `GET /api/v2/reporting/agencies/{toptier_code}/{fiscal_year}/{fiscal_period}/unlinked_awards/{assistance|procurement}/`
- Was a submission on time: `GET /api/v2/reporting/agencies/{toptier_code}/{fiscal_year}/{fiscal_period}/submission_history/`
  compared against that period's `submission_due_date` from `GET /api/v2/references/submission_periods/`.
- Path parameters go straight into the path: `.../049/2024/12/...`, not `.../fiscal_year/2024/...`.

## Computing an unlinked percentage
unlinked % = (unlinked_file_c_award_count + unlinked_file_d_award_count)
             / (those two + total_linked_award_count), all from the unlinked_awards endpoint.
Never divide by award counts from another endpoint (e.g. /api/v2/agency/awards/count/) - those
count a different population and give a meaningless ratio.
File C is award-level data from agency financial systems; File D is award records from
FPDS/FABS. "Unlinked" means a record in one has no match in the other.

## Known limitations
- The API gives unlinked award COUNTS only, never dollar amounts. If asked "how much money" is
  unlinked, say that plainly and offer the counts instead.
- `obligation_difference` is the File A vs. File B obligation difference, not unlinked dollars.
- `tas_obligation_not_in_gtas_total` is obligations in GTAS for accounts missing from File A.

verified: contracts@03b9e255, live API 2026-10-14

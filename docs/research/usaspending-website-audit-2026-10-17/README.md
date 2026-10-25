# USAspending website audit (2026-10-17)

A list of 163 things a person can do on usaspending.gov (search awards, open a profile, explore budgets, download data), each tied to the API request the website sends for it. The point is to measure the chat agent against the website: what it can reach, where its numbers could differ, and what to build next.

## How it was checked

1. **Website code.** Every page traced to the helper that calls the API, at fedspendingtransparency/usaspending-website `d550b3e110` (API contracts at `0a6d86a1e7`).
2. **Live replay.** 156/156 runnable requests returned HTTP 2xx; each request and response is saved under `evidence/`. The other 7 are export jobs or the website's key-gated AI search, traced but not run.
3. **Browser capture.** Playwright loaded 21 live pages and ran 2 click-through flows, logging every API call. 113 tasks were seen this way; 44 need typing, a click, or a submit and are traced from code only; 6 use a helper the website never calls.

Baseline: FY2025 = 2024-10-01 through 2025-09-30. Account queries use FY2025 Q4 / period 12 unless specified. Directory and disaster totals have their own documented scopes.

## What matters for this repo

1. **District rankings lead with a catch-all bucket.** For California FY2025 award obligations by congressional district, the top row is `CA-MULTIPLE DISTRICTS` at $203.3B; the largest real district, CA-36, has $7.8B (Q102). Nothing in the backend separates it out.
2. **One search page shows two measures.** The Advanced Search results table uses `spending_level: "awards"` (lifetime award value); the time, category, and map charts beside it use `"transactions"` (obligations dated in the period). `spending_by_award` rejects `"transactions"`. NASA's top FY2025 contract shows $22.65B lifetime (Q001); all NASA FY2025 contract obligations total $15.78B (Q028). Answers need to name their measure (#230).
3. **IDV amounts depend on vehicle type.** NASA's largest IDIQs (80JSC019C0012, NNJ16GU21B, NNJ16GX08B) report zero child awards; the money is on the vehicle. SEWP V GWAC NNG15SE16B shows a $0 award amount with 43 child orders carrying the obligations (Q006, Q073).
4. **Downloads take a different agency ID.** `list_monthly_files`, `bulk_download/awards`, and `download/accounts` take `toptier_agency_id` (NASA 72) from `bulk_download/list_agencies`, not the `agency_id` (NASA 862) from `/agency/080/` (Q136, Q140).
5. **Cheap download helpers.** `/download/count/` sizes an export in under a second (21,273 transactions for NASA FY2025 contracts, limit 500,000; Q141). `/bulk_download/list_monthly_files/` lists pre-built per-agency, per-year archives (Q140).
6. **Website links need the website's filter format.** `/references/filter/` stores any JSON, but the site only opens hashes holding its own Redux state (`version: "2020-06-01"`) and sends anything else to `/hash-error`. `browser_capture_interactive.json` has a real payload to copy for "see this on USAspending.gov" links.
7. **Federal account pages use v1.** `/api/v1/tas/balances/quarters/total/` (16 calls per page load) and `/api/v1/tas/categories/total/` drive the account charts and have no `api_contracts` entry. The v2 account endpoints all worked live (Q105, Q107, Q108).
8. **Coverage.** 83 tasks hit an endpoint the backend already calls; 23 are website plumbing; 6 are dead on the site; 1 is the website's AI search; 50 are gaps. The chat-relevant gaps are disaster breakdowns (#110), federal/Treasury accounts (#303, #184), IDV child awards, award funding accounts, and state profiles (#119).

Smaller notes: state profiles mix windows (top-5 tables use a trailing 12 months, header totals the last full FY, and the % of total divides one by the other; the API's `year=latest` for a state is the in-progress FY); the state "Awards" top-5 is always contracts ranked by lifetime award amount, labeled Obligations; the state DEFC chart uses `spending_level: "award_financial"`; the site never calls `spending_by_transaction_grouped`, `autocomplete/psc`, `autocomplete/glossary`, `awards/last_updated`, `agency/{code}/sub_agency/count`, or `agency/{code}/federal_account/count`.

## Upstream quirks (don't affect this repo)

- **Quoted award IDs return 503.** The contract says a double-quoted ID in `award_ids` forces an exact match; live, every quoted request returned an empty HTTP 503. The backend and the website both send IDs unquoted (Q024 notes).
- **A 422 message mislabels two award-type groups.** Award search accepts types from one group per request. The error listing the groups names 06/10 `other_financial_assistance` and 09/11/-1 `direct_payments`, because `raise_if_award_types_not_valid_subset` in `usaspending_api/common/helpers/api_helper.py` passes those two mappings in the wrong order. The check itself is correct (Q021).

## Files

- `inventory.json`: every task with request, expected result, website source location, contract link, UI call sites, and `evidence_basis`.
- `evidence/Q###.json`: the exact live request and response for each executed task.
- `browser_capture.json`, `browser_capture_interactive.json`: API calls observed from 21 live pages and two click-through flows.
- `build_inventory.py`: regenerates `inventory.json` (expects upstream checkouts under `/tmp/usaspending-website-audit-20261017/`).
- `replay.py --phase N`: replays tasks; phases 2 and 3 use IDs extracted into `context.json` by earlier phases.
- `render_report.py`: regenerates this file.


## Advanced search: award questions and filters

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q001 | Which NASA contracts active in FY2025 have the largest award amounts? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: THE BOEING COMPANY, Award Amount $22.65B |
| Q002 | Which NASA grants active in FY2025 have the largest award amounts? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: BAYLOR COLLEGE OF MEDICINE, Award Amount $139.59M |
| Q003 | Which loans active in FY2025 have the largest reported loan values? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: TSMC ARIZONA CORP, Loan Value $1.00B |
| Q004 | Which direct-payment awards active in FY2025 have the largest award amounts? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: MULTIPLE RECIPIENTS, Award Amount $17.49B |
| Q005 | Which other financial-assistance awards active in FY2025 have the largest award amounts? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: INTERNATIONAL BANK FOR RECONSTRUCTION AND DEVELOPMENT, Award Amount $27.51B |
| Q006 | Which NASA government-wide acquisition contracts (GWACs) match FY2025? | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: BLUE TECH INC., Award Amount 0.0 |
| Q007 | Find FY2025 contracts matching the keyword cybersecurity. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: LOCKHEED MARTIN CORPORATION, Award Amount $666.32M |
| Q008 | Find FY2025 contracts awarded by Health and Human Services. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: LEIDOS BIOMEDICAL RESEARCH INC, Award Amount $4.81B |
| Q009 | Find FY2025 contracts funded by NASA, regardless of awarding agency. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: THE BOEING COMPANY, Award Amount $22.65B |
| Q010 | Find FY2025 contracts with place of performance in California. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, Award Amount $41.61B |
| Q011 | Find FY2025 contracts awarded to recipients located in California. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, Award Amount $41.61B |
| Q012 | Find FY2025 contracts performed outside the United States. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: FLUOR INTERCONTINENTAL, INC, Award Amount $12.57B |
| Q013 | Find FY2025 contracts for NAICS 541512, computer systems design. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: ORACLE HEALTH GOVERNMENT SERVICES, INC., Award Amount $2.12B |
| Q014 | Find FY2025 contracts with product/service code D302. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: L3HARRIS TECHNOLOGIES, INC., Award Amount $1.83B |
| Q015 | Find FY2025 contracts set aside for small businesses. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: T-REX SOLUTIONS LLC, Award Amount $1.48B |
| Q016 | Find FY2025 firm-fixed-price contracts. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: THE BOEING COMPANY, Award Amount $18.11B |
| Q017 | Find FY2025 contracts awarded using full and open competition. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: HUMANA GOVERNMENT BUSINESS INC, Award Amount $51.27B |
| Q018 | Find FY2025 contract awards between $1 million and $10 million. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: GRENLAR-VNW JV1, LLC, Award Amount $10.00M |
| Q019 | Find FY2025 contracts for recipients matching Lockheed Martin. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: LOCKHEED MARTIN CORP, Award Amount $48.06B |
| Q020 | Find FY2025 awards under Assistance Listing 93.575. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: DEPARTMENT OF EDUCATION CALIFORNIA, Award Amount $2.31B |
| Q021 | Find FY2025 grants tagged with Infrastructure Investment and Jobs Act DEF code Z. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: NATIONAL RAILROAD PASSENGER CORPORATION, Award Amount $15.64B |
| Q022 | Find FY2025 contracts for woman-owned businesses. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: M1 SUPPORT SERVICES, L.P., Award Amount $4.70B |
| Q023 | Find FY2025 contracts charged to NASA Treasury accounts. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: THE BOEING COMPANY, Award Amount $22.65B |
| Q024 | Retrieve a known contract by its award identifier. | `POST /v2/search/spending_by_award/` | browser | 1 rows; top: THE BOEING COMPANY, Award Amount $22.65B |

## Advanced search: totals, charts, maps, and grouped views

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q025 | How many FY2025 NASA awards match each award-type tab? | `POST /v2/search/spending_by_award_count/` | browser | contracts=9,912, direct_payments=0, grants=8,389, idvs=946, loans=0, other=0 |
| Q026 | How did NASA contract transaction obligations change by month in FY2025? | `POST /v2/search/spending_over_time/` | browser | 12 rows; top: aggregated_amount $1.19B |
| Q027 | How did NASA contract transaction obligations change by quarter in FY2025? | `POST /v2/search/spending_over_time/` | browser | 4 rows; top: aggregated_amount $2.64B |
| Q028 | How did NASA contract transaction obligations change by fiscal year in FY2025? | `POST /v2/search/spending_over_time/` | browser | 1 rows; top: aggregated_amount $15.78B |
| Q029 | Which awarding agency ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/awarding_agency/` | browser | 3 rows; top: Department of Defense, amount $491.81B |
| Q030 | Which funding agency ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/funding_agency/` | source | 3 rows; top: Department of Defense, amount $503.86B |
| Q031 | Which awarding subagency ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/awarding_subagency/` | browser | 3 rows; top: Department of the Navy, amount $176.57B |
| Q032 | Which funding subagency ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/funding_subagency/` | source | 3 rows; top: Department of the Navy, amount $185.23B |
| Q033 | Which recipient ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/recipient/` | browser | 3 rows; top: LOCKHEED MARTIN CORPORATION, amount $34.10B |
| Q034 | Which NAICS industry ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/naics/` | browser | 3 rows; top: Aircraft Manufacturing, amount $61.24B |
| Q035 | Which product/service code ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/psc/` | browser | 3 rows; top: AIRCRAFT, FIXED WING, amount $42.26B |
| Q036 | Which federal account ranks highest in FY2025 contract transaction obligations? | `POST /v2/search/spending_by_category/federal_account/` | browser | 3 rows; top: Defense Environmental Cleanup, Environmental and Other Defen, amount $44.09B |
| Q037 | Which Assistance Listings rank highest in FY2025 grant transaction obligations? | `POST /v2/search/spending_by_category/cfda/` | browser | 3 rows; top: Grants to States for Medicaid, amount $666.13B |
| Q038 | Which states had the most FY2025 NASA contract work performed there? | `POST /v2/search/spending_by_geography/` | browser | 51 rows; top: California, aggregated_amount $4.28B |
| Q039 | Which states were home to recipients of FY2025 NASA contract obligations? | `POST /v2/search/spending_by_geography/` | browser | 51 rows; top: California, aggregated_amount $4.75B |
| Q040 | How are FY2025 NASA contract obligations distributed across counties? | `POST /v2/search/spending_by_geography/` | browser | 501 rows; top: Los Angeles County, aggregated_amount $3.66B |
| Q041 | How are FY2025 NASA contract obligations distributed across congressional districts? | `POST /v2/search/spending_by_geography/` | browser | 398 rows; top: CA-28, aggregated_amount $2.38B |
| Q042 | Group FY2025 NASA contract transactions by prime award and rank the obligation totals. | `POST /v2/search/spending_by_transaction_grouped/` | dead code | 3 rows; top: transaction_obligation $22.65B |
| Q043 | Which NASA prime contracts have the largest reported subaward totals in the FY2025 search? | `POST /v2/search/spending_by_subaward_grouped/` | source | 3 rows; top: subaward_obligation $12.34B |
| Q044 | List individual NASA contract subawards matching FY2025. | `POST /v2/search/spending_by_award/` | browser | 3 rows; top: PARAGON SPACE DEVELOPMENT CORPORATION, Sub-Award Amount $180.74M |
| Q045 | Create a shareable URL for the NASA FY2025 contract search. | `POST /v2/references/filter/` | browser | hash=2986f5ef9c9900a71840d8016a3905d6 |
| Q046 | Restore the filters from that shareable search URL. | `POST /v2/references/hash/` | browser | filter |

## Search controls and reference lookups

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q047 | Find the NASA awarding-agency filter option. | `POST /v2/autocomplete/awarding_agency/` | source | 2 rows; top: id=862, toptier_flag=True |
| Q048 | Find the NASA funding-agency filter option. | `POST /v2/autocomplete/funding_agency/` | source | 2 rows; top: id=862, toptier_flag=True |
| Q049 | Find recipient filter options for Lockheed. | `POST /v2/autocomplete/recipient/` | source | 3 rows; top: LOCKHEED MARTIN CORPORATION |
| Q050 | Find location filter options matching California. | `POST /v2/autocomplete/location/` | source | cities=[{'city_name': 'CALIFORNIA', 'country_name': 'PERU'}, {'city_name': 'CALIFORNIA', 'country_name': 'GUATEMALA'}, {'city_name': 'CALIFORNIA', 'country_name |
| Q051 | Find Assistance Listings related to child care. | `POST /v2/autocomplete/cfda/` | source | 3 rows; top: program_number=66.444, program_title=Voluntary School and Child Care Lead Testing and Reduction Gran |
| Q052 | Find NAICS codes related to computers. | `POST /v2/autocomplete/naics/` | source | 3 rows; top: naics=334113, naics_description=Computer Terminal Manufacturing |
| Q053 | Find product/service codes related to software. | `POST /v2/autocomplete/psc/` | dead code | 3 rows; top: product_or_service_code=70, psc_description=INFORMATION TECHNOLOGY EQUIPMENT (INCLD FIRMWARE) SOFTWA |
| Q054 | Find program activities with research in the name. | `POST /v2/autocomplete/program_activity/` | source | 3 rows; top: program_activity_code=0000, program_activity_name=DATA ACT RESEARCH/UNDISTRIBUTED |
| Q055 | Browse the top-level NAICS industry hierarchy. | `GET /v2/references/naics/` | source | 24 rows; top: count 64 |
| Q056 | Expand professional, scientific, and technical services in the NAICS tree. | `GET /v2/references/naics/54/` | source | 1 rows; top: count 52 |
| Q057 | Browse the top level of the product/service-code tree. | `GET /v2/references/filter_tree/psc/` | source | 3 rows; top: count 890 |
| Q058 | Browse agencies in the Treasury-account filter tree. | `GET /v2/references/filter_tree/tas/` | source | 92 rows; top: Department of Agriculture (USDA), count 1,084 |
| Q059 | Expand NASA in the Treasury-account filter tree. | `GET /v2/references/filter_tree/tas/080/` | source | 16 rows; top: Office of Inspector General, National Aeronautics and Space , count 26 |
| Q060 | Populate state and congressional-district selection metadata. | `GET /v2/references/states/` | browser | 56 rows; top: ALASKA |

## Individual award profiles

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q061 | Open the selected NASA contract and inspect its recipient, agencies, dates, and financial summary. | `GET /v2/awards/{{contract_id}}/` | browser | id=311267295; generated_unique_award_id=CONT_AWD_NAS1510000_8000_-NONE-_-NONE-; piid=NAS1510000; category=contract |
| Q062 | Open the selected NASA grant and inspect its recipient, agencies, dates, and financial summary. | `GET /v2/awards/{{grant_id}}/` | browser | id=269004917; generated_unique_award_id=ASST_NON_NNX16AO69A_080; fain=NNX16AO69A; category=grant |
| Q063 | Open the selected loan and inspect its recipient, agencies, dates, and financial summary. | `GET /v2/awards/{{loan_id}}/` | browser | id=271536977; generated_unique_award_id=ASST_NON_TSMC2_013; fain=TSMC2; uri=TSMC2_0 |
| Q064 | What modifications and obligation changes occurred on the selected NASA contract? | `POST /v2/transactions/` | browser | 3 rows; top: INTERNATIONAL SPACE STATION, federal_action_obligation 0.0 |
| Q065 | Which subawards are linked to the selected NASA contract? | `POST /v2/subawards/` | source | 3 rows; top: WILDWOOD ELECTRONICS INC, amount $34.75K |
| Q066 | How many transactions does this NASA contract have? | `GET /v2/awards/count/transaction/{{contract_id}}/` | browser | transactions=1,656 |
| Q067 | How many subawards does this NASA contract have? | `GET /v2/awards/count/subaward/{{contract_id}}/` | browser | subawards=269 |
| Q068 | How many federal accounts fund this NASA contract? | `GET /v2/awards/count/federal_account/{{contract_id}}/` | browser | federal_accounts=1,222 |
| Q069 | Which federal accounts fund this NASA contract? | `POST /v2/awards/accounts/` | browser | 3 rows; top: total_transaction_obligated_amount=569247514.28, federal_account=080-0124 |
| Q070 | Summarize the agencies and accounts funding this NASA contract. | `POST /v2/awards/funding_rollup/` | browser | total_transaction_obligated_amount=$4.09B; awarding_agency_count=1; funding_agency_count=1; federal_account_count=4 |
| Q071 | Inspect the account-level funding records behind this NASA contract. | `POST /v2/awards/funding/` | source | 3 rows; top: transaction_obligated_amount=3300000.0, gross_outlay_amount=None |

## Indefinite-delivery vehicle profiles

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q072 | Open the selected NASA IDV and inspect its vehicle details. | `GET /v2/awards/{{idv_id}}/` | browser | id=351431442; generated_unique_award_id=CONT_IDV_NNG15SE16B_8000; piid=NNG15SE16B; category=idv |
| Q073 | What amounts are associated with the child and grandchild awards of this IDV? | `GET /v2/idvs/amounts/{{idv_id}}/` | browser | award_id=351431442; generated_unique_award_id=CONT_IDV_NNG15SE16B_8000; child_idv_count=0; child_award_count=43 |
| Q074 | List the child awards under this IDV. | `POST /v2/idvs/awards/` | browser | 3 rows; top: REVEAL FARSIGHT SOFTWARE LICENSE - RSO, obligated_amount $22.55K |
| Q075 | List the grandchild awards under this IDV. | `POST /v2/idvs/awards/` | browser | 0 rows (empty) |
| Q076 | Show the timeline of awards issued under this IDV. | `POST /v2/idvs/activity/` | browser | 3 rows of 43; top: BLUE TECH INC., obligated_amount $1.50M |
| Q077 | Which federal accounts fund awards under this IDV? | `POST /v2/idvs/accounts/` | browser | 3 rows; top: total_transaction_obligated_amount=16567.56, federal_account=080-0120 |
| Q078 | Summarize funding across this IDV and its related awards. | `POST /v2/idvs/funding_rollup/` | browser | total_transaction_obligated_amount=$1.59M; awarding_agency_count=8; funding_agency_count=8; federal_account_count=12 |
| Q079 | Inspect account funding records for this IDV family. | `POST /v2/idvs/funding/` | source | 3 rows; top: award_id=279032856, generated_unique_award_id=CONT_AWD_2033H622F00035_2036_NNG15SE16B_8000 |

## Agency directory and agency profiles

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q080 | Browse federal agencies and resolve NASA to its agency identifiers. | `GET /v2/references/toptier_agencies/` | browser | 111 rows; top: 400 Years of African-American History Commission, obligated_amount 0.0 |
| Q081 | What is NASA and what does its FY2025 agency profile describe? | `GET /v2/agency/080/` | browser | fiscal_year=2025; toptier_code=080; name=National Aeronautics and Space Administration; abbreviation=NASA |
| Q082 | How have NASA budgetary resources, obligations, and outlays changed across fiscal years? | `GET /v2/agency/080/budgetary_resources/` | browser | toptier_code=080 |
| Q083 | How are NASA FY2025 award obligations split among award categories? | `GET /v2/agency/080/obligations_by_award_category/` | browser | 6 rows; top: aggregated_amount $15.78B |
| Q084 | Which NASA subagencies awarded the most in FY2025? | `GET /v2/agency/080/sub_agency/` | browser | 1 rows of 1; top: National Aeronautics and Space Administration, total_obligations $22.86B |
| Q085 | How many NASA subagencies have FY2025 award activity? | `GET /v2/agency/080/sub_agency/count/` | dead code | toptier_code=080; fiscal_year=2025; sub_agency_count=1; office_count=13 |
| Q086 | How many new awards did NASA make in FY2025? | `GET /v2/agency/080/awards/new/count/` | browser | toptier_code=080; fiscal_year=2025; agency_type=awarding; new_award_count=6,730 |
| Q087 | What are NASA FY2025 award obligations and transaction counts? | `GET /v2/agency/080/awards/` | browser | fiscal_year=2025; latest_action_date=2025-09-30T00:00:00; toptier_code=080; transaction_count=36,114 |
| Q088 | How does NASA FY2025 funding break down by bureau or component? | `GET /v2/agency/080/sub_components/` | browser | 1 rows of 1; top: National Aeronautics and Space Administration, total_obligations $28.77B |
| Q089 | Drill into the first NASA component to list its FY2025 federal accounts. | `GET /v2/agency/080/sub_components/{{bureau_slug}}/` | source | 10 rows of 14; top: Exploration, National Aeronautics and Space Administration, total_obligations $7.92B |
| Q090 | Count federal accounts represented in NASA FY2025 agency spending. | `GET /v2/agency/080/federal_account/count/` | dead code | toptier_code=080; fiscal_year=2025; federal_account_count=13; treasury_account_count=125 |

## Recipient directory and recipient profiles

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q091 | Find Lockheed Martin in the recipient directory and resolve its profile ID. | `POST /v2/recipient/` | browser | 3 rows of 47; top: LOCKHEED MARTIN CORPORATION, amount $15.18B |
| Q092 | What did the selected Lockheed Martin recipient receive in FY2025? | `GET /v2/recipient/{{recipient_id}}/` | browser | name=LOCKHEED MARTIN CORPORATION; duns=008016958; uei=G4KDGE4JFFK7; recipient_id=6cf5fb1b-4988-d087-5dc1-70939d8fc6c4-C |
| Q093 | What did the selected Lockheed Martin recipient receive in all available years? | `GET /v2/recipient/{{recipient_id}}/` | browser | name=LOCKHEED MARTIN CORPORATION; duns=008016958; uei=G4KDGE4JFFK7; recipient_id=6cf5fb1b-4988-d087-5dc1-70939d8fc6c4-C |
| Q094 | Which child recipients roll up to the selected Lockheed Martin parent? | `GET /v2/recipient/children/{{recipient_parent_uei}}/` | source | 32 rows; first: LOCKHEED MARTIN CORPORATION, amount $2.20B |
| Q095 | How many new awards did this recipient receive each quarter in FY2025? | `POST /v2/search/new_awards_over_time/` | browser | 4 rows; top: new_award_count_in_period=71, time_period={'fiscal_year': '2025', 'quarter': '1'} |
| Q096 | Which agencies supplied this recipient with the most FY2025 obligations? | `POST /v2/search/spending_by_category/awarding_agency/` | browser | 3 rows; top: Department of Defense, amount $34.09B |
| Q097 | Which federal accounts supplied this recipient with FY2025 obligations? | `POST /v2/search/spending_by_category/federal_account/` | browser | 3 rows; top: Operation and Maintenance, Air Force, amount $555.02M |

## State profiles

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q098 | Browse the state profile directory. | `GET /v2/recipient/state/` | source | 56 rows; first: North Dakota, amount $11.54M |
| Q099 | How much federal award funding is associated with California in FY2025? | `GET /v2/recipient/state/06/` | browser | name=California; code=CA; fips=06; type=state |
| Q100 | How is California FY2025 award funding divided among award types? | `GET /v2/recipient/state/awards/06/` | browser | 5 rows; first: amount $62.04B |
| Q101 | Which California counties have the most FY2025 award transaction obligations? | `POST /v2/search/spending_by_category/county/` | browser | 3 rows; top: LOS ANGELES COUNTY, amount $80.12B |
| Q102 | Which California congressional districts have the most FY2025 award transaction obligations? | `POST /v2/search/spending_by_category/district/` | browser | 3 rows; top: CA-MULTIPLE DISTRICTS, amount $203.34B |

## Federal accounts and Treasury accounts

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q103 | Which NASA federal accounts have the largest FY2025 budgetary resources? | `POST /v2/federal_accounts/` | browser | 3 rows; top: budgetary_resources $15.54B |
| Q104 | Open the selected NASA federal account and list its Treasury accounts. | `GET /v2/federal_accounts/{{account_code}}/` | browser | 10 rows; top: Exploration, National Aeronautics and Space Administration, obligated_amount $128.26K |
| Q105 | What are the selected account’s FY2025 budgetary resources, obligations, and outlays? | `GET /v2/federal_accounts/{{account_id}}/fiscal_year_snapshot/2025/` | browser | outlay=$7.67B, budget_authority=$15.54B, obligated=$7.92B, unobligated=$7.62B, balance_brought_forward=453296422.29, other_budgetary_resources=0.0, appropriatio |
| Q106 | Which object-class filters are available for this federal account? | `GET /v2/federal_accounts/{{account_id}}/available_object_classes/` | browser | 5 rows; top: Acquisition of assets |
| Q107 | Break down the selected Treasury account’s FY2025 spending by program activity. | `GET /v2/agency/treasury_account/{{tas}}/program_activity/` | source | 1 rows of 1; top: DEEP SPACE EXPLORATION SYSTEMS (DIRECT), obligated_amount $128.26K |
| Q108 | Break down the selected Treasury account’s FY2025 spending by object class. | `GET /v2/agency/treasury_account/{{tas}}/object_class/` | source | 1 rows of 1; top: Contractual services and supplies, obligated_amount $128.26K |
| Q109 | Which certified fiscal periods can be selected for account views? | `GET /v2/references/submission_periods/` | browser | available_periods |

## Spending Explorer

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q110 | Which agencies had the most account obligations through FY2025 Q4? | `POST /v2/spending/` | browser | 110 rows; top: Department of Health and Human Services, amount $2.79T |
| Q111 | How were account obligations divided across budget functions through FY2025 Q4? | `POST /v2/spending/` | browser | 20 rows; top: Medicare, amount $1.84T |
| Q112 | What did the government spend on by object class through FY2025 Q4? | `POST /v2/spending/` | browser | 7 rows; top: Grants and fixed charges, amount $7.39T |
| Q113 | Drill from NASA to its federal accounts in Spending Explorer. | `POST /v2/spending/` | browser | 14 rows; top: Exploration, National Aeronautics and Space Administration, amount $7.92B |
| Q114 | Drill from general science, space, and technology to budget subfunctions. | `POST /v2/spending/` | browser | 2 rows; top: Space flight, research, and supporting activities, amount $27.81B |
| Q115 | Drill from the selected federal account to program activities. | `POST /v2/spending/` | browser | 2 rows; top: DEEP SPACE EXPLORATION SYSTEMS (DIRECT), amount $7.92B |
| Q116 | Which recipients appear under the selected account in Spending Explorer? | `POST /v2/spending/` | browser | 675 rows; top: THE BOEING COMPANY, amount $1.17B |

## Disaster and emergency funding

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q117 | Which emergency-funding laws and DEF codes can be selected? | `GET /v2/references/def_codes/` | browser | 52 rows; top: Infrastructure Investment and Jobs Act |
| Q118 | What are the overall COVID-19 funding resources, obligations, and outlays? | `GET /v2/disaster/overview/` | browser | total_budget_authority=$4.68T |
| Q119 | How much COVID-19 funding is associated with awards? | `POST /v2/disaster/award/amount/` | browser | award_count=23,587,782; obligation=$2.25T; outlay=$2.30T; face_value_of_loan=$1.20T |
| Q120 | How many agencies received COVID-19 emergency funding? | `POST /v2/disaster/agency/count/` | browser | count=48 |
| Q121 | Which agencies account for the most COVID-19 award obligations? | `POST /v2/disaster/agency/spending/` | browser | 3 rows of 48; top: Small Business Administration, obligation $744.02B |
| Q122 | Which Assistance Listings account for the most COVID-19 award obligations? | `POST /v2/disaster/cfda/spending/` | browser | 3 rows of 540; top: Paycheck Protection Loan Program (PPP), obligation $811.47B |
| Q123 | Which recipients account for the most COVID-19 award obligations? | `POST /v2/disaster/recipient/spending/` | source | 3 rows of 10,000; top: DEPARTMENT OF EDUCATION CALIFORNIA, obligation $30.12B |
| Q124 | Which federal accounts account for the most COVID-19 award obligations? | `POST /v2/disaster/federal_account/spending/` | source | 281 rows of 281; top: Business Loans Program Account, Small Business Administratio, obligation $813.19B |
| Q125 | Which object classes account for the most COVID-19 award obligations? | `POST /v2/disaster/object_class/spending/` | source | 5 rows of 5; top: Grants and fixed charges, obligation $2.21T |
| Q126 | Which agencies issued the largest COVID-19 loan face values? | `POST /v2/disaster/agency/loans/` | source | 3 rows of 4; top: Small Business Administration, obligation $857.68B |
| Q127 | Map COVID-19 award spending by recipient state. | `POST /v2/disaster/spending_by_geography/` | browser | 58 rows; top: California, amount $280.56B |

## Data quality and agency submission statistics

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q128 | What total budgetary resources underpin FY2025 period 12 reporting statistics? | `GET /v2/references/total_budgetary_resources/` | browser | 1 rows; top: total_budgetary_resources $13.26T |
| Q129 | Compare agencies’ FY2025 period 12 submission statistics. | `GET /v2/reporting/agencies/overview/` | browser | 3 rows of 111; top: Department of Health and Human Services |
| Q130 | When did agencies publish their FY2025 submissions? | `GET /v2/reporting/agencies/publish_dates/` | source | 3 rows of 111; top: Department of Health and Human Services |
| Q131 | Inspect NASA’s reporting history and submission metrics. | `GET /v2/reporting/agencies/080/overview/` | browser | 3 rows of 84; top: total_budgetary_resources $15.77T |
| Q132 | When was NASA’s FY2025 period 12 submission certified or republished? | `GET /v2/reporting/agencies/080/2025/12/submission_history/` | source | 3 rows of 3; top: publication_date=2025-12-29T19:43:57.059633+00:00, certification_date=2025-12-29T19:43:57.097431+00: |
| Q133 | Which NASA account balances are missing or discrepant in FY2025 period 12? | `GET /v2/reporting/agencies/080/discrepancies/` | source | 0 rows (empty) |
| Q134 | Where do NASA account and award obligations differ in FY2025 period 12? | `GET /v2/reporting/agencies/080/differences/` | source | 3 rows of 13; top: 080-X-0130-000 |
| Q135 | How many NASA procurement records fail award linkage in FY2025 period 12? | `GET /v2/reporting/agencies/080/2025/12/unlinked_awards/procurement/` | source | unlinked_file_c_award_count=36; unlinked_file_d_award_count=164; total_linked_award_count=2,511 |

## Downloads and exports

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q136 | Which agencies can I choose for a custom award download? | `POST /v2/bulk_download/list_agencies/` | browser | agencies, sub_agencies |
| Q137 | Which agencies can I choose for an account-data download? | `POST /v2/bulk_download/list_agencies/` | browser | agencies, sub_agencies |
| Q138 | Which budget functions are available for an account download? | `GET /v2/budget_functions/list_budget_functions/` | browser | 20 rows; top: budget_function_code=750, budget_function_title=Administration of Justice |
| Q139 | Which budget subfunctions fall under general science, space, and technology? | `POST /v2/budget_functions/list_budget_subfunctions/` | source | 2 rows; top: budget_subfunction_code=251, budget_subfunction_title=General science and basic research |
| Q140 | Find pre-generated NASA FY2025 contract download archives. | `POST /v2/bulk_download/list_monthly_files/` | browser | 2 rows; top: National Aeronautics and Space Administration |
| Q141 | Estimate the transaction count before downloading the NASA FY2025 search. | `POST /v2/download/count/` | source | calculated_transaction_count=21,273; maximum_transaction_limit=500,000; calculated_count=21,273; spending_level=transactions |
| Q142 | Download the selected contract profile’s data. | `POST /v2/download/contract/` | source | not run: Source/contract traced. Export generation was not run; no completed ZIP is claimed. |
| Q143 | Download the selected grant profile’s data. | `POST /v2/download/assistance/` | source | not run: Source/contract traced. Export generation was not run; no completed ZIP is claimed. |
| Q144 | Download the selected IDV profile’s data. | `POST /v2/download/idv/` | source | not run: Source/contract traced. Export generation was not run; no completed ZIP is claimed. |
| Q145 | Export the NASA FY2025 search as award, transaction, and subaward CSV files. | `POST /v2/download/search/` | source | not run: Source/contract traced; potentially large export not generated. |
| Q146 | Generate a custom NASA contract transaction download for October 2024. | `POST /v2/bulk_download/awards/` | source | not run: Source/contract traced; bulk generation not executed. |
| Q147 | Download NASA FY2025 period 12 account balances. | `POST /v2/download/accounts/` | source | not run: Source/contract traced; account export not generated. |

## Definitions, data freshness, and natural-language search

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q148 | When was award data last updated? | `GET /v2/awards/last_updated/` | dead code | last_updated=10/17/2026 |
| Q149 | Load the glossary for the website’s glossary panel. | `GET /v2/references/glossary/` | browser | 151 rows; top: Submission Period |
| Q150 | Look up the definition of obligation. | `POST /v2/autocomplete/glossary/` | dead code | 3 rows; top: Deobligation |
| Q151 | Inspect the data dictionary to understand fields in downloaded files. | `GET /v2/references/data_dictionary/` | browser | document |
| Q152 | Ask the website’s optional natural-language search for NASA contracts in California. | `POST /v2/llm/filter-search/` | source | not run: Conditional feature: separate LLM_API configuration and X-LLM-API-Key header; SearchContainer gates access. Availability and output not verified in production. |

## Calls observed in the browser but missing from the source trace

| ID | Question | Call | Basis | Live result |
|---|---|---|---|---|
| Q153 | How many federal accounts fund awards under this IDV? | `GET /v2/idvs/count/federal_account/{{idv_id}}/` | browser | count=77 |
| Q154 | In which countries was work performed for this recipient’s FY2025 awards? | `POST /v2/search/spending_by_category/country/` | browser | 5 rows; top: UNITED STATES, amount $34.09B |
| Q155 | In which states or territories was work performed for this recipient’s FY2025 awards? | `POST /v2/search/spending_by_category/state_territory/` | browser | 5 rows; top: Texas, amount $30.48B |
| Q156 | Which emergency/disaster funding laws (DEF codes) supported California awards in FY2025? | `POST /v2/search/spending_by_category/defc/` | browser | 5 rows; top: American Rescue Plan Act of 2021, amount $20.84B |
| Q157 | How have the selected federal account’s outlays moved quarter by quarter? | `POST /v1/tas/balances/quarters/total/` | browser | 39 rows; top: item=2017, aggregate=2027056039.46 |
| Q158 | Which program activities carry the selected federal account’s obligations? | `POST /v1/tas/categories/total/` | browser | 3 rows; top: item=EXPLORATION (DIRECT), aggregate=33591666034.81 |
| Q159 | How many recipients received COVID-19 award funding? | `POST /v2/disaster/recipient/count/` | browser | count=14,671,865 |
| Q160 | How many Assistance Listings funded COVID-19 grants? | `POST /v2/disaster/cfda/count/` | browser | count=335 |
| Q161 | Keyword search: summarize all transactions matching cybersecurity. | `POST /v2/search/transaction_spending_summary/` | source | prime_awards_count=30,897, prime_awards_obligation_amount=$29.97B |
| Q162 | Keyword search: how many matching transactions fall in each award-type tab? | `POST /v2/search/spending_by_transaction_count/` | source | contracts=22,194, grants=6,282, idvs=1,807, loans=116, direct_payments=447, other=51 |
| Q163 | Keyword search: list the largest contract transactions matching cybersecurity. | `POST /v2/search/spending_by_transaction/` | source | 3 rows; top: MIRACLE SYSTEMS LLC, Transaction Amount $88.76M |

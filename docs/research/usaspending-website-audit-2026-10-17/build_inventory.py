import json
import re
from pathlib import Path

ROOT = Path('/Users/jameswagner/Projects/USSpendingRag/docs/research/usaspending-website-audit-2026-10-17')
TMP = Path('/tmp/usaspending-website-audit-20261017')
WEB = TMP / 'usaspending-website-master'
API = TMP / 'usaspending-api-master'
ROOT.mkdir(parents=True, exist_ok=True)
website_sha = json.loads((TMP / 'website-head.json').read_text())['sha']
api_sha = json.loads((TMP / 'api-head.json').read_text())['sha']
rows = []
section = ''
page = '/search'
CONTRACTS = ['A', 'B', 'C', 'D']
GRANTS = ['02', '03', '04', '05']
IDVS = ['IDV_A', 'IDV_B', 'IDV_B_A', 'IDV_B_B', 'IDV_B_C', 'IDV_C', 'IDV_D', 'IDV_E']
ALL = CONTRACTS + GRANTS + ['06', '07', '08', '09', '10', '11', '-1']
ASSISTANCE = GRANTS + ['06', '07', '08', '09', '10', '11', '-1']
FY = [{'start_date': '2024-10-01', 'end_date': '2025-09-30'}]
NASA = {'type': 'awarding', 'tier': 'toptier', 'name': 'National Aeronautics and Space Administration'}
BASE = {'award_type_codes': CONTRACTS, 'time_period': FY}
NF = {**BASE, 'agencies': [NASA]}
fields = ['Award ID', 'Recipient Name', 'Award Amount', 'Start Date', 'End Date', 'Description', 'generated_internal_id']

def source(file, symbol):
    p = WEB / 'src/js' / file
    lines = p.read_text().splitlines()
    hits = [i+1 for i, line in enumerate(lines) if re.search(r'(?:const|function)\s+' + re.escape(symbol) + r'\b', line)]
    if not hits:
        hits = [i+1 for i, line in enumerate(lines) if symbol in line]
    if not hits:
        raise ValueError((file, symbol))
    return {'path': 'src/js/' + file, 'symbol': symbol, 'line': hits[0], 'url': f'https://github.com/fedspendingtransparency/usaspending-website/blob/{website_sha}/src/js/{file}#L{hits[0]}'}

def add(question, endpoint, data, expected, file, symbol, *, method='POST', contract=None, phase=1, run=True, notes='', depends=(), extract=None, dead=False):
    if contract is None:
        contract = endpoint.split('?')[0].strip('/').removeprefix('api/') + '.md'
    cp = API / 'usaspending_api/api_contracts/contracts' / contract
    row = {'id': f'Q{len(rows)+1:03}', 'section': section, 'question': question,
           'website_page': 'https://www.usaspending.gov' + page,
           'request': {'method': method, 'path': endpoint, 'query' if method == 'GET' else 'body': data},
           'expected_result': expected, 'source': source(file, symbol),
           'contract': {'path': contract, 'url': f'https://github.com/fedspendingtransparency/usaspending-api/blob/{api_sha}/usaspending_api/api_contracts/contracts/{contract}'} if cp.exists() else None,
           'phase': phase, 'execute': run, 'notes': notes, 'depends_on': list(depends), 'ui_dead_code': dead}
    if extract: row['extract_context'] = extract
    rows.append(row)
    return row['id']

def search(question, filters=None, types=None, usefields=None, **kw):
    f = {**BASE, **(filters or {})}
    if types: f['award_type_codes'] = types
    fs = usefields or fields
    return add(question, '/api/v2/search/spending_by_award/', {'filters': f, 'fields': fs, 'limit': 3, 'page': 1, 'sort': 'Award Amount' if 'Award Amount' in fs else fs[0], 'order': 'desc', 'spending_level': 'awards'}, 'Matching award rows with requested fields and page metadata; Award Amount is an award-level amount, not the sum of FY2025 transactions.', 'helpers/searchHelper.js', 'performSpendingByAwardSearch', **kw)

section = 'Advanced search: award questions and filters'
contract_search = search('Which NASA contracts active in FY2025 have the largest award amounts?', {'agencies': [NASA]}, extract={'contract_id': 'results.0.generated_internal_id', 'contract_piid': 'results.0.Award ID'})
grant_search = search('Which NASA grants active in FY2025 have the largest award amounts?', {'agencies': [NASA]}, types=GRANTS, extract={'grant_id': 'results.0.generated_internal_id'})
loan_search = search('Which loans active in FY2025 have the largest reported loan values?', types=['07', '08'], usefields=['Award ID', 'Recipient Name', 'Loan Value', 'Subsidy Cost', 'generated_internal_id'], extract={'loan_id': 'results.0.generated_internal_id'})
search('Which direct-payment awards active in FY2025 have the largest award amounts?', types=['06', '10'])
search('Which other financial-assistance awards active in FY2025 have the largest award amounts?', types=['09', '11', '-1'])
idv_search = search('Which NASA government-wide acquisition contracts (GWACs) match FY2025?', {'agencies': [NASA]}, types=['IDV_A'], notes='GWACs chosen so child-award views return data: the top NASA IDIQs by award amount (e.g. 80JSC019C0012, NNJ16GU21B, NNJ16GX08B) report zero child and grandchild awards in /idvs/amounts/.', usefields=['Award ID', 'Recipient Name', 'Award Amount', 'Last Date to Order', 'generated_internal_id'], extract={'idv_id': 'results.0.generated_internal_id'})
search('Find FY2025 contracts matching the keyword cybersecurity.', {'keywords': ['cybersecurity']})
search('Find FY2025 contracts awarded by Health and Human Services.', {'agencies': [{'type': 'awarding', 'tier': 'toptier', 'name': 'Department of Health and Human Services'}]})
search('Find FY2025 contracts funded by NASA, regardless of awarding agency.', {'agencies': [{**NASA, 'type': 'funding'}]})
search('Find FY2025 contracts with place of performance in California.', {'place_of_performance_locations': [{'country': 'USA', 'state': 'CA'}]})
search('Find FY2025 contracts awarded to recipients located in California.', {'recipient_locations': [{'country': 'USA', 'state': 'CA'}]})
search('Find FY2025 contracts performed outside the United States.', {'place_of_performance_scope': 'foreign'})
search('Find FY2025 contracts for NAICS 541512, computer systems design.', {'naics_codes': {'require': ['541512']}})
search('Find FY2025 contracts with product/service code D302.', {'psc_codes': ['D302']})
search('Find FY2025 contracts set aside for small businesses.', {'set_aside_type_codes': ['SBA']})
search('Find FY2025 firm-fixed-price contracts.', {'contract_pricing_type_codes': ['J']})
search('Find FY2025 contracts awarded using full and open competition.', {'extent_competed_type_codes': ['A']})
search('Find FY2025 contract awards between $1 million and $10 million.', {'award_amounts': [{'lower_bound': 1000000, 'upper_bound': 10000000}]})
search('Find FY2025 contracts for recipients matching Lockheed Martin.', {'recipient_search_text': ['LOCKHEED MARTIN']})
search('Find FY2025 awards under Assistance Listing 93.575.', {'program_numbers': ['93.575']}, types=GRANTS)
search('Find FY2025 grants tagged with Infrastructure Investment and Jobs Act DEF code Z.', {'def_codes': ['Z']}, types=GRANTS, notes='spending_by_award accepts award types from one group only; the website issues one query per tab.')
search('Find FY2025 contracts for woman-owned businesses.', {'recipient_type_names': ['woman_owned_business']})
search('Find FY2025 contracts charged to NASA Treasury accounts.', {'tas_codes': {'require': [['080']]}})
search('Retrieve a known contract by its award identifier.', {'award_ids': ['{{contract_piid}}']}, phase=2, depends=[contract_search], notes='Website sends the uppercased ID unquoted (AwardIDSearch.jsx). The contract-documented double-quoted exact-match form returned an empty-body HTTP 503 on every live attempt (2026-10-17), with and without time_period.')

section = 'Advanced search: totals, charts, maps, and grouped views'
add('How many FY2025 NASA awards match each award-type tab?', '/api/v2/search/spending_by_award_count/', {'filters': {**NF, 'award_type_codes': ALL + IDVS}, 'subawards': False}, 'Counts for contracts, IDVs, grants, loans, direct payments, and other assistance.', 'helpers/searchHelper.js', 'performSpendingByAwardTabCountSearch')
for group, label in [('month', 'month'), ('quarter', 'quarter'), ('fiscal_year', 'fiscal year')]:
    add(f'How did NASA contract transaction obligations change by {label} in FY2025?', '/api/v2/search/spending_over_time/', {'filters': NF, 'group': group, 'spending_level': 'transactions'}, 'Time buckets with aggregated obligations and any outlay fields returned; transaction dates define the selected period.', 'helpers/searchHelper.js', 'performSpendingOverTimeSearch')
for cat, label in [('awarding_agency', 'awarding agency'), ('funding_agency', 'funding agency'), ('awarding_subagency', 'awarding subagency'), ('funding_subagency', 'funding subagency'), ('recipient', 'recipient'), ('naics', 'NAICS industry'), ('psc', 'product/service code'), ('federal_account', 'federal account')]:
    add(f'Which {label} ranks highest in FY2025 contract transaction obligations?', f'/api/v2/search/spending_by_category/{cat}/', {'filters': BASE, 'category': cat, 'limit': 3, 'page': 1, 'spending_level': 'transactions'}, 'Ranked category rows with identifiers, names, amounts, and pagination; sample contains the first three rows.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch', contract=f'v2/search/spending_by_category/{cat}.md')
add('Which Assistance Listings rank highest in FY2025 grant transaction obligations?', '/api/v2/search/spending_by_category/cfda/', {'filters': {**BASE, 'award_type_codes': GRANTS}, 'category': 'cfda', 'limit': 3, 'page': 1, 'spending_level': 'transactions'}, 'Assistance Listing numbers/names and aggregate grant obligations.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch')
for scope, layer, question in [('place_of_performance', 'state', 'Which states had the most FY2025 NASA contract work performed there?'), ('recipient_location', 'state', 'Which states were home to recipients of FY2025 NASA contract obligations?'), ('place_of_performance', 'county', 'How are FY2025 NASA contract obligations distributed across counties?'), ('place_of_performance', 'district', 'How are FY2025 NASA contract obligations distributed across congressional districts?')]:
    add(question, '/api/v2/search/spending_by_geography/', {'filters': NF, 'scope': scope, 'geo_layer': layer, 'spending_level': 'transactions'}, 'Geographic shapes/identifiers and aggregated amounts; recipient location and place of performance answer different questions.', 'helpers/searchHelper.js', 'performSpendingByGeographySearch')
add('Group FY2025 NASA contract transactions by prime award and rank the obligation totals.', '/api/v2/search/spending_by_transaction_grouped/', {'filters': NF, 'limit': 3, 'page': 1, 'sort': 'transaction_obligation', 'order': 'desc'}, 'Award ID, transaction_count, transaction_obligation, and award_generated_internal_id.', 'helpers/searchHelper.js', 'performSpendingByTransactionsGrouped', dead=True, notes='Website helper has no call site in the current source; endpoint works but the site never calls it. The Advanced Search table only offers awards/subawards (useResultsTableSearch.jsx); spending_by_award rejects spending_level=transactions live.')
add('Which NASA prime contracts have the largest reported subaward totals in the FY2025 search?', '/api/v2/search/spending_by_subaward_grouped/', {'filters': NF, 'limit': 3, 'page': 1, 'sort': 'subaward_obligation', 'order': 'desc'}, 'Prime award, subaward count and obligation, prime obligation, and subaward-to-prime ratio.', 'helpers/searchHelper.js', 'performSpendingBySubawardGrouped')
add('List individual NASA contract subawards matching FY2025.', '/api/v2/search/spending_by_award/', {'filters': NF, 'spending_level': 'subawards', 'subawards': True, 'fields': ['Sub-Award ID', 'Sub-Awardee Name', 'Sub-Award Amount', 'Sub-Award Date', 'Prime Award ID'], 'limit': 3, 'page': 1, 'sort': 'Sub-Award Amount', 'order': 'desc'}, 'Individual subaward rows, not prime awards; subaward date and amount use a different basis.', 'helpers/searchHelper.js', 'performSpendingByAwardSearch')
share = add('Create a shareable URL for the NASA FY2025 contract search.', '/api/v2/references/filter/', {'filters': NF}, 'A hash token for an arbitrary JSON filter object.', 'helpers/searchHelper.js', 'generateUrlHash', extract={'filter_hash': 'hash'}, notes='The API stores any JSON, but the website stores its own Redux filter state ({filters: {timePeriodType, selectedRecipients, ...}, version: "2020-06-01"}) and parseRemoteFilters rejects anything else. Opening /search?hash=<this hash> in a browser redirected to /hash-error; see browser_capture_interactive.json for a website-generated payload.')
add('Restore the filters from that shareable search URL.', '/api/v2/references/hash/', {'hash': '{{filter_hash}}'}, 'The saved filter object, suitable for repopulating the search controls.', 'helpers/searchHelper.js', 'restoreUrlHash', phase=2, depends=[share])

section = 'Search controls and reference lookups'
for endpoint, symbol, text, question, expected in [
    ('awarding_agency', 'fetchAwardingAgencies', 'NASA', 'Find the NASA awarding-agency filter option.', 'Matching top-tier and sub-tier agencies with identifiers and names.'),
    ('funding_agency', 'fetchFundingAgencies', 'NASA', 'Find the NASA funding-agency filter option.', 'Matching funding agencies with identifiers and names.'),
    ('recipient', 'fetchRecipientsAutocomplete', 'LOCKHEED', 'Find recipient filter options for Lockheed.', 'Recipient name, UEI/DUNS, and identifiers where available.'),
    ('location', 'fetchLocations', 'California', 'Find location filter options matching California.', 'Matching location objects and structured geographic identifiers.'),
    ('cfda', 'fetchCFDA', 'child care', 'Find Assistance Listings related to child care.', 'Matching program numbers and titles.'),
    ('naics', 'fetchNAICS', 'computer', 'Find NAICS codes related to computers.', 'Matching NAICS codes and descriptions.'),
    ('psc', 'fetchPSC', 'software', 'Find product/service codes related to software.', 'Matching PSC codes and descriptions.'),
    ('program_activity', 'fetchProgramActivity', 'research', 'Find program activities with research in the name.', 'Matching program activity codes and names.')]:
    add(question, f'/api/v2/autocomplete/{endpoint}/', {'search_text': text, 'limit': 3}, expected, 'helpers/searchHelper.js', symbol)
for path, symbol, question, expected in [
    ('references/naics', 'naicsRequest', 'Browse the top-level NAICS industry hierarchy.', 'Two-digit NAICS groups and child availability.'),
    ('references/naics/54', 'naicsRequest', 'Expand professional, scientific, and technical services in the NAICS tree.', 'NAICS 54 and its immediate children.'),
    ('references/filter_tree/psc', 'fetchPsc', 'Browse the top level of the product/service-code tree.', 'PSC root groups with child counts or descriptions.'),
    ('references/filter_tree/tas', 'fetchTas', 'Browse agencies in the Treasury-account filter tree.', 'Agency identifiers with federal-account children.'),
    ('references/filter_tree/tas/080', 'fetchTas', 'Expand NASA in the Treasury-account filter tree.', 'NASA federal-account identifiers and names.')]:
    ct = {'references/naics/54': 'v2/references/naics/naics_code.md', 'references/filter_tree/tas/080': 'v2/references/filter_tree/tas/agency.md'}.get(path)
    add(question, f'/api/v2/{path}/', {}, expected, 'helpers/searchHelper.js', symbol, method='GET', contract=ct)
add('Populate state and congressional-district selection metadata.', '/api/v2/references/states/', {}, 'States, districts, and territories used by the location controls.', 'hooks/useStateData.jsx', 'useStateData', method='GET')

section = 'Individual award profiles'
page = '/award/{{contract_id}}'
for token, label, dep in [('contract_id', 'NASA contract', contract_search), ('grant_id', 'NASA grant', grant_search), ('loan_id', 'loan', loan_search)]:
    add(f'Open the selected {label} and inspect its recipient, agencies, dates, and financial summary.', f'/api/v2/awards/{{{{{token}}}}}/', {}, 'Award detail, recipient and agency objects, period of performance, and type-specific amounts. Loans expose face value and subsidy cost.', 'helpers/searchHelper.js', 'fetchAwardV2', method='GET', contract='v2/awards/award_id.md', phase=2, depends=[dep])
add('What modifications and obligation changes occurred on the selected NASA contract?', '/api/v2/transactions/', {'award_id': '{{contract_id}}', 'limit': 3, 'page': 1, 'sort': 'action_date', 'order': 'desc'}, 'Transaction dates, modifications, action types, descriptions, and federal_action_obligation.', 'helpers/searchHelper.js', 'fetchAwardTransaction', phase=2, depends=[contract_search])
add('Which subawards are linked to the selected NASA contract?', '/api/v2/subawards/', {'award_id': '{{contract_id}}', 'limit': 3, 'page': 1}, 'Subaward recipient, amount, date, and description; an empty result is valid.', 'helpers/searchHelper.js', 'performSubawardSearch', phase=2, depends=[contract_search])
for typ, question in [('transaction', 'How many transactions does this NASA contract have?'), ('subaward', 'How many subawards does this NASA contract have?'), ('federal_account', 'How many federal accounts fund this NASA contract?')]:
    add(question, f'/api/v2/awards/count/{typ}/{{{{contract_id}}}}/', {}, 'A count used in the award-history tab label.', 'helpers/awardHistoryHelper.js', 'getAwardHistoryCounts', method='GET', contract=f'v2/awards/count/{typ}/award_id.md', phase=2, depends=[contract_search])
for path, symbol, question, expected in [('accounts', 'fetchAwardFederalAccounts', 'Which federal accounts fund this NASA contract?', 'Account code/title, agency, and total transaction obligated amount.'), ('funding_rollup', 'fetchAwardFundingSummary', 'Summarize the agencies and accounts funding this NASA contract.', 'Funding rollup counts and obligations.'), ('funding', 'fetchFederalAccountFunding', 'Inspect the account-level funding records behind this NASA contract.', 'Reporting period, account, program activity, object class, obligations, and outlays.')]:
    add(question, f'/api/v2/awards/{path}/', {'award_id': '{{contract_id}}', **({} if path == 'funding_rollup' else {'limit': 3, 'page': 1})}, expected, 'helpers/awardHistoryHelper.js' if path == 'funding' else 'helpers/awardSummaryHelper.js', symbol, phase=2, depends=[contract_search])

section = 'Indefinite-delivery vehicle profiles'
page = '/award/{{idv_id}}'
add('Open the selected NASA IDV and inspect its vehicle details.', '/api/v2/awards/{{idv_id}}/', {}, 'IDV identifiers, recipient, parent relationships, dates, and contract details.', 'helpers/searchHelper.js', 'fetchAwardV2', method='GET', contract='v2/awards/award_id.md', phase=2, depends=[idv_search])
add('What amounts are associated with the child and grandchild awards of this IDV?', '/api/v2/idvs/amounts/{{idv_id}}/', {}, 'Separate child/grandchild award counts and obligation/award amounts.', 'helpers/idvHelper.js', 'fetchAwardAmounts', method='GET', contract='v2/idvs/amounts/award_id.md', phase=2, depends=[idv_search])
for typ in ['child_awards', 'grandchild_awards']:
    add(f'List the {typ.replace("_", " ")} under this IDV.', '/api/v2/idvs/awards/', {'award_id': '{{idv_id}}', 'type': typ, 'limit': 3, 'page': 1}, 'Related award IDs, descriptions, agencies, dates, and obligated amounts.', 'helpers/idvHelper.js', 'fetchReferencedAwards', phase=2, depends=[idv_search])
for path, symbol, question, expected in [('activity', 'fetchIdvActivity', 'Show the timeline of awards issued under this IDV.', 'Child/grandchild award periods and amounts for the activity visualization.'), ('accounts', 'fetchIdvFederalAccounts', 'Which federal accounts fund awards under this IDV?', 'Federal account codes, titles, agencies, and obligation totals.'), ('funding_rollup', 'fetchIdvFundingSummary', 'Summarize funding across this IDV and its related awards.', 'Funding-agency and account counts, and total obligations.'), ('funding', 'fetchAwardFedAccountFunding', 'Inspect account funding records for this IDV family.', 'File C funding rows with accounts, periods, obligations, and outlays.')]:
    add(question, f'/api/v2/idvs/{path}/', {'award_id': '{{idv_id}}', **({} if path == 'funding_rollup' else {'limit': 3, 'page': 1})}, expected, 'helpers/idvHelper.js', symbol, phase=2, depends=[idv_search])

section = 'Agency directory and agency profiles'
page = '/agency/national-aeronautics-and-space-administration'
agency_list = add('Browse federal agencies and resolve NASA to its agency identifiers.', '/api/v2/references/toptier_agencies/', {}, 'Agency names, slugs, top-tier codes, internal agency IDs, and budgetary amounts.', 'helpers/agencyLandingHelper.js', 'fetchAllAgencies', method='GET')
add('What is NASA and what does its FY2025 agency profile describe?', '/api/v2/agency/080/', {'fiscal_year': 2025}, 'Agency name, mission, website, ID, abbreviation, and subagency count.', 'apis/agency.js', 'fetchAgencyOverview', method='GET', contract='v2/agency/toptier_code.md', extract={'agency_id': 'agency_id'})
add('How have NASA budgetary resources, obligations, and outlays changed across fiscal years?', '/api/v2/agency/080/budgetary_resources/', {}, 'Fiscal-year budgetary-resource records and agency share of government resources.', 'apis/agency.js', 'fetchBudgetaryResources', method='GET', contract='v2/agency/toptier_code/budgetary_resources.md')
for tail, symbol, question, expected in [
    ('obligations_by_award_category', 'fetchObligationsByAwardType', 'How are NASA FY2025 award obligations split among award categories?', 'Contract, grant, loan, and other award-category obligations.'),
    ('sub_agency', 'fetchSubagencySpendingList', 'Which NASA subagencies awarded the most in FY2025?', 'Subagency obligations, transaction counts, new awards, and child offices.'),
    ('sub_agency/count', 'fetchSubagencyCount', 'How many NASA subagencies have FY2025 award activity?', 'Subagency count for the selected year and optional award types.'),
    ('awards/new/count', 'fetchSubagencyNewAwardsCount', 'How many new awards did NASA make in FY2025?', 'Number of new awards, distinct from the number of transactions.'),
    ('awards', 'fetchSubagencySummary', 'What are NASA FY2025 award obligations and transaction counts?', 'Agency award-summary totals for the selected year.'),
    ('sub_components', 'fetchSubcomponentsList', 'How does NASA FY2025 funding break down by bureau or component?', 'Bureau/component identifiers, budgetary resources, obligations, and outlays.')]:
    add(question, f'/api/v2/agency/080/{tail}/', {'fiscal_year': 2025}, expected, 'apis/agency.js', symbol, method='GET', contract=f'v2/agency/toptier_code/{tail}.md', extract={'bureau_slug': 'results.0.id'} if tail=='sub_components' else None)
bureau_parent = rows[-1]['id']
add('Drill into the first NASA component to list its FY2025 federal accounts.', '/api/v2/agency/080/sub_components/{{bureau_slug}}/', {'fiscal_year': 2025}, 'Federal-account identifiers, names, resources, obligations, and outlays.', 'apis/agency.js', 'fetchFederalAccountsList', method='GET', contract='v2/agency/toptier_code/sub_components/bureau_slug.md', phase=2, depends=[bureau_parent])
add('Count federal accounts represented in NASA FY2025 agency spending.', '/api/v2/agency/080/federal_account/count/', {'fiscal_year': 2025}, 'Federal-account count for the agency and year.', 'apis/agency.js', 'fetchSpendingCount', method='GET', contract='v2/agency/toptier_code/federal_account/count.md')

section = 'Recipient directory and recipient profiles'
page = '/recipient'
recipient = add('Find Lockheed Martin in the recipient directory and resolve its profile ID.', '/api/v2/recipient/', {'keyword': 'LOCKHEED MARTIN CORPORATION', 'award_type': 'contracts', 'sort': 'amount', 'order': 'desc', 'limit': 3, 'page': 1}, 'Recipient name, hash-plus-level ID, UEI/DUNS, level, and amount; directory query has no fiscal-year filter.', 'helpers/recipientLandingHelper.js', 'fetchAllRecipients', extract={'recipient_id': 'results.0.id', 'recipient_uei': 'results.0.uei'})
for year, desc in [('2025', 'FY2025'), ('all', 'all available years')]:
    add(f'What did the selected Lockheed Martin recipient receive in {desc}?', '/api/v2/recipient/{{recipient_id}}/', {'year': year}, 'Recipient identity/location, business categories, award counts, and award-type amounts.', 'helpers/recipientHelper.js', 'fetchRecipientOverview', method='GET', contract='v2/recipient/recipient_id.md', phase=2, depends=[recipient], extract={'recipient_parent_uei': 'parent_uei'} if year == '2025' else None)
add('Which child recipients roll up to the selected Lockheed Martin parent?', '/api/v2/recipient/children/{{recipient_parent_uei}}/', {'year': 2025}, 'Child recipient IDs, names, UEIs/DUNS, and amounts; the route requires the parent UEI, not a child-level UEI.', 'helpers/recipientHelper.js', 'fetchChildRecipients', method='GET', contract='v2/recipient/children/duns_or_uei.md', phase=3, depends=[rows[-2]['id']])
add('How many new awards did this recipient receive each quarter in FY2025?', '/api/v2/search/new_awards_over_time/', {'filters': {'recipient_id': '{{recipient_id}}', 'time_period': FY}, 'group': 'quarter'}, 'Quarterly new-award counts by award category.', 'helpers/recipientHelper.js', 'fetchNewAwardCounts', phase=2, depends=[recipient])
for cat, question in [('awarding_agency', 'Which agencies supplied this recipient with the most FY2025 obligations?'), ('federal_account', 'Which federal accounts supplied this recipient with FY2025 obligations?')]:
    add(question, f'/api/v2/search/spending_by_category/{cat}/', {'filters': {**BASE, 'recipient_id': '{{recipient_id}}'}, 'category': cat, 'limit': 3, 'page': 1, 'spending_level': 'transactions'}, 'Ranked categories and obligation amounts restricted to the resolved recipient ID.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch', phase=2, depends=[recipient])

section = 'State profiles'
page = '/state/california'
add('Browse the state profile directory.', '/api/v2/recipient/state/', {}, 'States and territories with FIPS codes and profile identifiers.', 'apis/state.js', 'fetchStateList', method='GET')
add('How much federal award funding is associated with California in FY2025?', '/api/v2/recipient/state/06/', {'year': 2025}, 'State summary with total award amount, recipient/award counts, population, and contextual measures.', 'apis/state.js', 'fetchStateOverview', method='GET', contract='v2/recipient/state/fips.md')
add('How is California FY2025 award funding divided among award types?', '/api/v2/recipient/state/awards/06/', {'year': 2025}, 'Award-category totals and counts for the state.', 'apis/state.js', 'fetchAwardBreakdown', method='GET', contract='v2/recipient/state/awards/fips.md')
for cat, label in [('county', 'counties'), ('district', 'congressional districts')]:
    add(f'Which California {label} have the most FY2025 award transaction obligations?', f'/api/v2/search/spending_by_category/{cat}/', {'filters': {'award_type_codes': ALL, 'time_period': FY, 'place_of_performance_locations': [{'country': 'USA', 'state': 'CA'}]}, 'category': cat, 'limit': 3, 'page': 1, 'spending_level': 'transactions'}, 'Ranked geographic categories with obligation totals in the selected state scope.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch')

section = 'Federal accounts and Treasury accounts'
page = '/federal_account'
account = add('Which NASA federal accounts have the largest FY2025 budgetary resources?', '/api/v2/federal_accounts/', {'filters': {'agency_identifier': '080', 'fy': '2025'}, 'sort': {'field': 'budgetary_resources', 'direction': 'desc'}, 'limit': 3, 'page': 1}, 'Federal-account number, database ID, account title, managing agency, and resources.', 'helpers/accountLandingHelper.js', 'fetchAllAccounts', extract={'account_code': 'results.0.account_number', 'account_id': 'results.0.account_id'})
account_detail = add('Open the selected NASA federal account and list its Treasury accounts.', '/api/v2/federal_accounts/{{account_code}}/', {'fiscal_year': 2025}, 'Account identity and associated Treasury-account records.', 'apis/account.js', 'fetchFederalAccount', method='GET', contract='v2/federal_accounts/account_number.md', phase=2, depends=[account], extract={'tas': 'children.0.code'})
add('What are the selected account’s FY2025 budgetary resources, obligations, and outlays?', '/api/v2/federal_accounts/{{account_id}}/fiscal_year_snapshot/2025/', {}, 'Account fiscal-year financial snapshot; this route uses the numeric account ID.', 'apis/account.js', 'fetchFederalAccountFYSnapshot', method='GET', contract='v2/federal_accounts/federal_account_id/fiscal_year_snapshot.md', phase=2, depends=[account])
add('Which object-class filters are available for this federal account?', '/api/v2/federal_accounts/{{account_id}}/available_object_classes/', {}, 'Available major/minor object-class codes and names.', 'apis/account.js', 'fetchAvailableObjectClasses', method='GET', contract='v2/federal_accounts/federal_account_id/available_object_classes.md', phase=2, depends=[account])
for tail, symbol, label in [('program_activity', 'fetchProgramActivityByTas', 'program activity'), ('object_class', 'fetchObjectClassByTas', 'object class')]:
    add(f'Break down the selected Treasury account’s FY2025 spending by {label}.', f'/api/v2/agency/treasury_account/{{{{tas}}}}/{tail}/', {'fiscal_year': 2025, 'page': 1}, 'Account obligations and outlays grouped by the selected classification.', 'apis/agency.js', symbol, method='GET', contract=f'v2/agency/treasury_account/tas/{tail}.md', phase=3, depends=[account_detail])
add('Which certified fiscal periods can be selected for account views?', '/api/v2/references/submission_periods/', {}, 'Submission fiscal years/months, reporting dates, and reveal dates.', 'apis/account.js', 'fetchAllSubmissionDates', method='GET')

section = 'Spending Explorer'
page = '/explorer'
for typ, question in [('agency', 'Which agencies had the most account obligations through FY2025 Q4?'), ('budget_function', 'How were account obligations divided across budget functions through FY2025 Q4?'), ('object_class', 'What did the government spend on by object class through FY2025 Q4?')]:
    add(question, '/api/v2/spending/', {'type': typ, 'filters': {'fy': '2025', 'quarter': '4'}}, 'Spending Explorer groups with IDs, names, values, and total; account obligations are cumulative through the selected quarter.', 'helpers/explorerHelper.js', 'fetchBreakdown')
for typ, extra, question in [('federal_account', {'agency': '{{agency_id}}'}, 'Drill from NASA to its federal accounts in Spending Explorer.'), ('budget_subfunction', {'budget_function': '250'}, 'Drill from general science, space, and technology to budget subfunctions.'), ('program_activity', {'federal_account': '{{account_id}}'}, 'Drill from the selected federal account to program activities.'), ('recipient', {'federal_account': '{{account_id}}'}, 'Which recipients appear under the selected account in Spending Explorer?')]:
    add(question, '/api/v2/spending/', {'type': typ, 'filters': {'fy': '2025', 'quarter': '4', **extra}}, 'Scoped Explorer children with identifiers, labels, amounts, and total.', 'helpers/explorerHelper.js', 'fetchBreakdown', phase=2, depends=[account] if 'federal_account' in extra else [])

section = 'Disaster and emergency funding'
page = '/disaster/covid-19'
add('Which emergency-funding laws and DEF codes can be selected?', '/api/v2/references/def_codes/', {}, 'DEF code, public law, title, and disaster grouping.', 'apis/disaster.js', 'fetchDEFCodes', method='GET')
add('What are the overall COVID-19 funding resources, obligations, and outlays?', '/api/v2/disaster/overview/', {'def_codes': 'L,M,N,O,P,U,V'}, 'Account-level supplemental funding overview for the selected laws.', 'apis/disaster.js', 'fetchOverview', method='GET')
DF = {'def_codes': ['L', 'M', 'N', 'O', 'P', 'U', 'V']}
add('How much COVID-19 funding is associated with awards?', '/api/v2/disaster/award/amount/', {'filter': {**DF, 'award_type_codes': ALL}}, 'Award-linked File C obligations/outlays and loan amounts, distinct from total account resources.', 'apis/disaster.js', 'fetchAwardAmounts')
add('How many agencies received COVID-19 emergency funding?', '/api/v2/disaster/agency/count/', {'filter': DF}, 'Count of agencies for the selected DEF codes.', 'apis/disaster.js', 'fetchAgencyCount')
for typ, label in [('agency', 'agencies'), ('cfda', 'Assistance Listings'), ('recipient', 'recipients'), ('federal_account', 'federal accounts'), ('object_class', 'object classes')]:
    symbol = {'agency': 'fetchAwardSpendingByAgency', 'cfda': 'fetchSpendingByCfda'}.get(typ, 'fetchDisasterSpending')
    add(f'Which {label} account for the most COVID-19 award obligations?', f'/api/v2/disaster/{typ}/spending/', {'filter': {**DF, 'award_type_codes': ASSISTANCE if typ == 'cfda' else ALL}, 'spending_type': 'award', 'pagination': {'limit': 3, 'page': 1, 'sort': 'obligation', 'order': 'desc'}}, 'Category rows containing obligations, outlays, award counts, and category identity.', 'apis/disaster.js', symbol)
add('Which agencies issued the largest COVID-19 loan face values?', '/api/v2/disaster/agency/loans/', {'filter': {**DF, 'award_type_codes': ['07', '08']}, 'pagination': {'limit': 3, 'page': 1, 'sort': 'face_value_of_loan', 'order': 'desc'}}, 'Loan face value and subsidy cost by agency; these are separate financial measures.', 'apis/disaster.js', 'fetchLoansByAgency')
add('Map COVID-19 award spending by recipient state.', '/api/v2/disaster/spending_by_geography/', {'filter': {**DF, 'award_type_codes': ALL}, 'geo_layer': 'state', 'scope': 'recipient_location', 'spending_type': 'obligation'}, 'Geographic identifiers with disaster-funded obligation/outlay amounts.', 'apis/disaster.js', 'fetchRecipientSpendingByGeography')

section = 'Data quality and agency submission statistics'
page = '/submission-statistics'
for endpoint, symbol, question, expected, query, contract in [
    ('references/total_budgetary_resources', 'getTotalBudgetaryResources', 'What total budgetary resources underpin FY2025 period 12 reporting statistics?', 'Government-wide budgetary resources for the selected reporting period.', {'fiscal_year': 2025, 'fiscal_period': 12}, None),
    ('reporting/agencies/overview', 'getAgenciesReportingData', 'Compare agencies’ FY2025 period 12 submission statistics.', 'Agency reporting totals, percentages, and linkage metrics.', {'fiscal_year': 2025, 'fiscal_period': 12, 'limit': 3, 'page': 1}, None),
    ('reporting/agencies/publish_dates', 'getSubmissionPublicationDates', 'When did agencies publish their FY2025 submissions?', 'Agency publication and certification dates by period.', {'fiscal_year': 2025, 'limit': 3, 'page': 1}, None),
    ('reporting/agencies/080/overview', 'fetchAgency', 'Inspect NASA’s reporting history and submission metrics.', 'NASA submission-period totals and reporting-quality metrics.', {'fiscal_year': 2025, 'limit': 3, 'page': 1}, 'v2/reporting/agencies/toptier_code/overview.md'),
    ('reporting/agencies/080/2025/12/submission_history', 'fetchPublishDates', 'When was NASA’s FY2025 period 12 submission certified or republished?', 'Submission history with publication/certification times and status.', {'limit': 3, 'page': 1}, 'v2/reporting/agencies/toptier_code/fiscal_year/fiscal_period/submission_history.md'),
    ('reporting/agencies/080/discrepancies', 'fetchMissingAccountBalances', 'Which NASA account balances are missing or discrepant in FY2025 period 12?', 'Treasury-account discrepancies between reference and submitted balances.', {'fiscal_year': 2025, 'fiscal_period': 12, 'limit': 3}, 'v2/reporting/agencies/toptier_code/discrepancies.md'),
    ('reporting/agencies/080/differences', 'fetchReportingDifferences', 'Where do NASA account and award obligations differ in FY2025 period 12?', 'Account-vs-award obligation differences by Treasury account.', {'fiscal_year': 2025, 'fiscal_period': 12, 'limit': 3}, 'v2/reporting/agencies/toptier_code/differences.md'),
    ('reporting/agencies/080/2025/12/unlinked_awards/procurement', 'fetchUnlinkedData', 'How many NASA procurement records fail award linkage in FY2025 period 12?', 'Linked/unlinked procurement-award record counts.', {}, 'v2/reporting/agencies/toptier_code/fiscal_year/fiscal_period/unlinked_awards/type.md')]:
    add(question, '/api/v2/' + endpoint + '/', query, expected, 'apis/agencyReporting.js', symbol, method='GET', contract=contract)

section = 'Downloads and exports'
page = '/download_center/custom_award_data'
add('Which agencies can I choose for a custom award download?', '/api/v2/bulk_download/list_agencies/', {'type': 'award_agencies'}, 'Selectable top-tier agencies and their toptier_agency_id values, which download endpoints use instead of the agency-profile agency_id.', 'helpers/bulkDownloadHelper.js', 'requestAgenciesList', extract={'toptier_agency_id': 'agencies.cfo_agencies.17.toptier_agency_id'})
add('Which agencies can I choose for an account-data download?', '/api/v2/bulk_download/list_agencies/', {'type': 'account_agencies'}, 'Agencies with account submissions.', 'helpers/bulkDownloadHelper.js', 'requestAgenciesList')
add('Which budget functions are available for an account download?', '/api/v2/budget_functions/list_budget_functions/', {}, 'Budget-function codes and titles.', 'helpers/bulkDownloadHelper.js', 'requestBudgetFunctionList', method='GET')
add('Which budget subfunctions fall under general science, space, and technology?', '/api/v2/budget_functions/list_budget_subfunctions/', {'budget_function': '250'}, 'Budget-subfunction codes/titles under function 250.', 'helpers/bulkDownloadHelper.js', 'requestBudgetSubfunctionList')
add('Find pre-generated NASA FY2025 contract download archives.', '/api/v2/bulk_download/list_monthly_files/', {'agency': '{{toptier_agency_id}}', 'fiscal_year': 2025, 'type': 'contracts'}, 'Archive filenames, download URLs, sizes, and update timestamps.', 'helpers/bulkDownloadHelper.js', 'requestArchiveFiles', phase=2)
add('Estimate the transaction count before downloading the NASA FY2025 search.', '/api/v2/download/count/', {'filters': NF}, 'Count of matching transactions to help size an export.', 'helpers/downloadHelper.js', 'requestDownloadCount')
for typ, symbol, token, label in [('contract', 'fetchContractDownloadFile', 'contract_id', 'contract'), ('assistance', 'fetchAssistanceDownloadFile', 'grant_id', 'grant'), ('idv', 'fetchIdvDownloadFile', 'idv_id', 'IDV')]:
    add(f'Download the selected {label} profile’s data.', f'/api/v2/download/{typ}/', {'award_id': '{{' + token + '}}'}, 'Download job/file metadata; poll /api/v2/download/status/ with the returned file_name and then fetch file_url.', 'helpers/downloadHelper.js', symbol, phase=2, run=False, notes='Source/contract traced. Export generation was not run; no completed ZIP is claimed.')
add('Export the NASA FY2025 search as award, transaction, and subaward CSV files.', '/api/v2/download/search/', {'filters': NF, 'spending_level': ['awards', 'transactions', 'subawards'], 'file_format': 'csv'}, 'Job metadata and eventual ZIP URL; status polling uses /api/v2/download/status/.', 'helpers/downloadHelper.js', 'requestFullDownloadNew', run=False, notes='Source/contract traced; potentially large export not generated.')
add('Generate a custom NASA contract transaction download for October 2024.', '/api/v2/bulk_download/awards/', {'filters': {'agency': '{{toptier_agency_id}}', 'prime_award_types': ['A', 'B', 'C', 'D'], 'date_type': 'action_date', 'date_range': {'start_date': '2024-10-01', 'end_date': '2024-10-31'}}, 'file_format': 'csv'}, 'Bulk export job metadata; poll /api/v2/bulk_download/status/ before retrieving the ZIP.', 'helpers/bulkDownloadHelper.js', 'requestAwardsDownload', phase=2, run=False, notes='Source/contract traced; bulk generation not executed.')
add('Download NASA FY2025 period 12 account balances.', '/api/v2/download/accounts/', {'filters': {'agency': '{{toptier_agency_id}}', 'fy': '2025', 'period': '12'}, 'account_level': 'treasury_account', 'submission_types': ['account_balances'], 'file_format': 'csv'}, 'Account export job metadata and eventual ZIP containing selected account data.', 'helpers/bulkDownloadHelper.js', 'requestAccountsDownload', phase=2, run=False, notes='Source/contract traced; account export not generated.')

section = 'Definitions, data freshness, and natural-language search'
page = '/'
add('When was award data last updated?', '/api/v2/awards/last_updated/', {}, 'Latest award-data update date for the website freshness indicator.', 'helpers/searchHelper.js', 'fetchLastUpdate', method='GET')
add('Load the glossary for the website’s glossary panel.', '/api/v2/references/glossary/', {'limit': 500}, 'Glossary terms, definitions, and related metadata.', 'helpers/glossaryHelper.js', 'fetchAllTerms', method='GET')
add('Look up the definition of obligation.', '/api/v2/autocomplete/glossary/', {'search_text': 'obligation', 'limit': 3}, 'Matching glossary terms with definitions.', 'helpers/glossaryHelper.js', 'fetchSearchResults')
add('Inspect the data dictionary to understand fields in downloaded files.', '/api/v2/references/data_dictionary/', {}, 'Data dictionary and crosswalk structure, including field definitions and source mappings.', 'apis/dataDictionary.js', 'fetchDataDictionary', method='GET')
add('Ask the website’s optional natural-language search for NASA contracts in California.', '/api/v2/llm/filter-search/', {'query': 'NASA contracts performed in California in fiscal year 2025'}, 'Streaming search interpretation/filter output; the UI then applies the resulting search filters.', 'components/search/collapsibleSidebar/useRequestNLSearch.jsx', 'useRequestNLSearch', contract='v2/llm/filter_search.md', run=False, notes='Conditional feature: separate LLM_API configuration and X-LLM-API-Key header; SearchContainer gates access. Availability and output not verified in production.')

section = 'Calls observed in the browser but missing from the source trace'
page = '/award/{{idv_id}}'
add('How many federal accounts fund awards under this IDV?', '/api/v2/idvs/count/federal_account/{{idv_id}}/', {}, 'Count used in the IDV federal-account tab label.', 'helpers/awardHistoryHelper.js', 'getAwardHistoryFederalAccountsIdv', method='GET', contract='v2/idvs/count/federal_account/award_id.md', phase=2, depends=[idv_search])
page = '/recipient/{{recipient_id}}/latest'
for cat, question in [('country', 'In which countries was work performed for this recipient’s FY2025 awards?'), ('state_territory', 'In which states or territories was work performed for this recipient’s FY2025 awards?')]:
    add(question, f'/api/v2/search/spending_by_category/{cat}/', {'filters': {'recipient_id': '{{recipient_id}}', 'time_period': FY}, 'category': cat, 'limit': 5, 'page': 1}, 'Ranked place-of-performance categories for the recipient. The live page defaults to a trailing 12 months, not a fiscal year.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch', contract=f'v2/search/spending_by_category/{cat}.md', phase=2, depends=[recipient])
page = '/state/california/latest'
add('Which emergency/disaster funding laws (DEF codes) supported California awards in FY2025?', '/api/v2/search/spending_by_category/defc/', {'filters': {'def_codes': ['L', 'M', 'N', 'O', 'P', 'U', 'V', 'Z', '1'], 'place_of_performance_scope': 'domestic', 'place_of_performance_locations': [{'country': 'USA', 'state': 'CA'}], 'time_period': FY}, 'category': 'defc', 'limit': 5, 'page': 1, 'spending_level': 'award_financial'}, 'DEF codes ranked by File C (award_financial) obligations for awards performed in the state.', 'helpers/searchHelper.js', 'performSpendingByCategorySearch', contract='v2/search/spending_by_category/defc.md')
page = '/federal_account/{{account_code}}'
add('How have the selected federal account’s outlays moved quarter by quarter?', '/api/v1/tas/balances/quarters/total/', {'filters': [{'field': 'treasury_account_identifier__federal_account_id', 'operation': 'equals', 'value': '{{account_id}}'}], 'group': ['submission__reporting_fiscal_year', 'submission__reporting_fiscal_quarter'], 'field': 'gross_outlay_amount_by_tas_cpe', 'aggregate': 'sum', 'order': ['submission__reporting_fiscal_year']}, 'Quarterly sums of a File A balance field. Legacy v1 generic-aggregation endpoint with no api_contracts entry; the account page calls it once per plotted series.', 'helpers/accountQuartersHelper.js', 'fetchTasBalanceTotals', phase=2, depends=[account])
add('Which program activities carry the selected federal account’s obligations?', '/api/v1/tas/categories/total/', {'group': 'program_activity__program_activity_name', 'field': 'obligations_incurred_by_program_object_class_cpe', 'aggregate': 'sum', 'order': ['-aggregate'], 'filters': [{'field': 'treasury_account__federal_account', 'operation': 'equals', 'value': '{{account_id}}'}], 'page': 1, 'limit': 5}, 'Program activities ranked by File B obligations. Legacy v1 endpoint with no api_contracts entry.', 'apis/account.js', 'fetchTasCategoryTotals', phase=2, depends=[account])
page = '/disaster/covid-19'
add('How many recipients received COVID-19 award funding?', '/api/v2/disaster/recipient/count/', {'filter': DF}, 'Recipient count for the selected DEF codes.', 'apis/disaster.js', 'fetchDisasterSpendingCount', contract='v2/disaster/recipient/count.md')
add('How many Assistance Listings funded COVID-19 grants?', '/api/v2/disaster/cfda/count/', {'filter': {**DF, 'award_type_codes': ['02', '03', '04', 'F001']}}, 'Assistance Listing count for the selected DEF codes and award-type tab.', 'apis/disaster.js', 'fetchCfdaCount', contract='v2/disaster/cfda/count.md')
page = '/keyword_search/cybersecurity'
KW = {'keyword': 'cybersecurity'}
add('Keyword search: summarize all transactions matching cybersecurity.', '/api/v2/search/transaction_spending_summary/', {'filters': KW}, 'Matching transaction count and total obligation, with no time filter.', 'helpers/keywordHelper.js', 'fetchSummary')
add('Keyword search: how many matching transactions fall in each award-type tab?', '/api/v2/search/spending_by_transaction_count/', {'filters': KW}, 'Transaction counts per award category for the keyword.', 'helpers/keywordHelper.js', 'performTabCountSearch')
add('Keyword search: list the largest contract transactions matching cybersecurity.', '/api/v2/search/spending_by_transaction/', {'filters': {**KW, 'award_type_codes': CONTRACTS}, 'fields': ['Award ID', 'Mod', 'Recipient Name', 'Action Date', 'Transaction Amount', 'Awarding Agency', 'Award Type', 'internal_id', 'generated_internal_id'], 'page': 1, 'limit': 3, 'sort': 'Transaction Amount', 'order': 'desc'}, 'Transaction-level rows (one per modification), the only transaction table on the site.', 'helpers/keywordHelper.js', 'performKeywordSearch')

DEAD_SYMBOLS = {'fetchPSC', 'fetchSubagencyCount', 'fetchSpendingCount', 'fetchLastUpdate', 'fetchSearchResults'}
for r in rows:
    if r['source']['symbol'] in DEAD_SYMBOLS:
        r['ui_dead_code'] = True
        r['notes'] = (r['notes'] + ' ' if r['notes'] else '') + 'Website helper has no call site in the current source; endpoint works but the site never calls it.'

all_source = [(p, p.read_text(errors='replace').splitlines()) for p in (WEB / 'src/js').rglob('*') if p.suffix in {'.js', '.jsx'}]
for r in rows:
    symbol = r['source']['symbol']
    callers = []
    for p, ls in all_source:
        if str(p.relative_to(WEB)) == r['source']['path']: continue
        hits = [i+1 for i,l in enumerate(ls) if re.search(r'\b' + re.escape(symbol) + r'\s*\(',l)]
        if hits:
            path = str(p.relative_to(WEB))
            callers.append({'path': path, 'line': hits[0], 'url': f'https://github.com/fedspendingtransparency/usaspending-website/blob/{website_sha}/{path}#L{hits[0]}'})
    r['ui_call_sites'] = callers[:5]

def _norm(path):
    path = path.rstrip('/') + '/'
    path = re.sub(r'/(CONT|ASST)_[^/]+/', '/{award}/', path)
    path = re.sub(r'/\d{6,}/', '/{award}/', path)
    path = re.sub(r'/[0-9a-f]{8}-[0-9a-f-]+-[CPR]/', '/{recipient}/', path)
    path = re.sub(r'/federal_accounts/\d+/', '/federal_accounts/{id}/', path)
    path = re.sub(r'/federal_accounts/\d{3}-\d{4}/', '/federal_accounts/{code}/', path)
    path = re.sub(r'/fiscal_year_snapshot/\d+/', '/fiscal_year_snapshot/{fy}/', path)
    path = re.sub(r'/sub_components/[a-z-]+/', '/sub_components/{bureau}/', path)
    return path

observed = {}
for name in ['browser_capture.json', 'browser_capture_interactive.json']:
    capture = ROOT / name
    if not capture.exists():
        continue
    data = json.loads(capture.read_text())
    for group in data.get('pages', []) + data.get('flows', []):
        for call in group['calls']:
            observed.setdefault(_norm(call['path']), set()).add(group.get('page') or group.get('flow'))
context = json.loads((ROOT / 'context.json').read_text())
for r in rows:
    path = re.sub(r'\{\{(\w+)\}\}', lambda m: str(context.get(m[1], m[0])), r['request']['path'])
    pages_seen = sorted(observed.get(_norm(path), []))
    r['browser_observed_on'] = pages_seen
    r['evidence_basis'] = 'browser_observed' if pages_seen else ('source_dead_code' if r['ui_dead_code'] else 'source_traced')

meta = {'audit_date': '2026-10-17', 'website_commit': website_sha, 'api_commit': api_sha,
        'api_base': 'https://api.usaspending.gov', 'endpoint_index': 'https://api.usaspending.gov/docs/endpoints',
        'method': 'Website source trace, direct public API replay of every executable task, and a Playwright (headless Chromium) capture of the API calls 21 live pages plus two click-through flows issue. evidence_basis says which applies per task.',
        'baseline': 'FY2025 = 2024-10-01 through 2025-09-30. Account queries use FY2025 Q4 / period 12 unless specified. Directory and disaster totals have their own documented scopes.'}
(ROOT / 'inventory.json').write_text(json.dumps({'metadata': meta, 'tasks': rows}, indent=2) + '\n')
print(f'Wrote {len(rows)} tasks; {sum(r["execute"] for r in rows)} executable; {sum(r["contract"] is None for r in rows)} without exact contract match')
for r in rows:
    if r['contract'] is None: print('CONTRACT', r['id'], r['request']['path'])
    if not r['ui_call_sites']: print('CALLER', r['id'], r['source']['symbol'])

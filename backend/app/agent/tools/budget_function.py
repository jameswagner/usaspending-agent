"""resolve_budget_function - name -> code lookup for get_spending_explorer_
breakdown's budget_function/budget_subfunction filters (issue #113).

That tool's own docstring says both filters take "a code, from a prior
call's result" - so today the only way to filter by, say, "Health" or
"National Defense" is to first call get_spending_explorer_breakdown with no
filter, browse the unfiltered list by eye, then make a second call with the
matching code. Same shape of gap resolve_naics_code/resolve_psc_code/
resolve_cfda_program/resolve_county_fips already closed elsewhere.

Unlike those, this vocabulary doesn't need a semantic retrieval index - it's
20 budget functions and 72 subfunctions, confirmed live 2026-09-19 via
GET /api/v2/budget_functions/list_budget_functions/ and (per-function)
POST /api/v2/budget_functions/list_budget_subfunctions/. Neither endpoint's
subfunction rows carry their parent function code, so the mapping below was
built by calling the subfunctions endpoint once per function code and
pairing each result set with the function it was filtered by.

Two budget function codes ("000" Governmental Receipts, "990" Multiple
functions) have no real subfunction breakdown - the live API still returns
one synthetic subfunction row each ("000"/"999") that just repeats or
renames the parent. Kept here for completeness since they're real, live
values, not placeholders.

Subfunction 601's title ("General retirement and disability insurance
(excluding social se...") is truncated exactly like this in the live API
response itself - not a copy error here.
"""
from __future__ import annotations

import difflib

from anthropic import beta_tool

from ._shared import _check_tool_call_budget, _record_tool_call, _wrap_untrusted

# code -> title
BUDGET_FUNCTIONS: dict[str, str] = {
    "000": "Governmental Receipts",
    "050": "National Defense",
    "150": "International Affairs",
    "250": "General Science, Space, and Technology",
    "270": "Energy",
    "300": "Natural Resources and Environment",
    "350": "Agriculture",
    "370": "Commerce and Housing Credit",
    "400": "Transportation",
    "450": "Community and Regional Development",
    "500": "Education, Training, Employment, and Social Services",
    "550": "Health",
    "570": "Medicare",
    "600": "Income Security",
    "650": "Social Security",
    "700": "Veterans Benefits and Services",
    "750": "Administration of Justice",
    "800": "General Government",
    "900": "Net Interest",
    "990": "Multiple functions",
}

# code -> (title, parent budget_function code)
BUDGET_SUBFUNCTIONS: dict[str, tuple[str, str]] = {
    "000": ("Governmental Receipts", "000"),
    "050": ("National Defense", "050"),
    "051": ("Department of Defense-Military", "050"),
    "053": ("Atomic energy defense activities", "050"),
    "054": ("Defense-related activities", "050"),
    "151": ("International development and humanitarian assistance", "150"),
    "152": ("International security assistance", "150"),
    "153": ("Conduct of foreign affairs", "150"),
    "154": ("Foreign information and exchange activities", "150"),
    "155": ("International financial programs", "150"),
    "251": ("General science and basic research", "250"),
    "252": ("Space flight, research, and supporting activities", "250"),
    "271": ("Energy supply", "270"),
    "272": ("Energy conservation", "270"),
    "274": ("Emergency energy preparedness", "270"),
    "276": ("Energy information, policy, and regulation", "270"),
    "301": ("Water resources", "300"),
    "302": ("Conservation and land management", "300"),
    "303": ("Recreational resources", "300"),
    "304": ("Pollution control and abatement", "300"),
    "306": ("Other natural resources", "300"),
    "351": ("Farm income stabilization", "350"),
    "352": ("Agricultural research and services", "350"),
    "371": ("Mortgage credit", "370"),
    "372": ("Postal service", "370"),
    "373": ("Deposit insurance", "370"),
    "376": ("Other advancement of commerce", "370"),
    "400": ("Transportation", "400"),
    "401": ("Ground transportation", "400"),
    "402": ("Air transportation", "400"),
    "403": ("Water transportation", "400"),
    "407": ("Other transportation", "400"),
    "451": ("Community development", "450"),
    "452": ("Area and regional development", "450"),
    "453": ("Disaster relief and insurance", "450"),
    "501": ("Elementary, secondary, and vocational education", "500"),
    "502": ("Higher education", "500"),
    "503": ("Research and general education aids", "500"),
    "504": ("Training and employment", "500"),
    "505": ("Other labor services", "500"),
    "506": ("Social services", "500"),
    "551": ("Health care services", "550"),
    "552": ("Health research and training", "550"),
    "554": ("Consumer and occupational health and safety", "550"),
    "571": ("Medicare", "570"),
    "601": ("General retirement and disability insurance (excluding social se", "600"),
    "602": ("Federal employee retirement and disability", "600"),
    "603": ("Unemployment compensation", "600"),
    "604": ("Housing assistance", "600"),
    "605": ("Food and nutrition assistance", "600"),
    "609": ("Other income security", "600"),
    "651": ("Social security", "650"),
    "701": ("Income security for veterans", "700"),
    "702": ("Veterans education, training, and rehabilitation", "700"),
    "703": ("Hospital and medical care for veterans", "700"),
    "704": ("Veterans housing", "700"),
    "705": ("Other veterans benefits and services", "700"),
    "751": ("Federal law enforcement activities", "750"),
    "752": ("Federal litigative and judicial activities", "750"),
    "753": ("Federal correctional activities", "750"),
    "754": ("Criminal justice assistance", "750"),
    "801": ("Legislative functions", "800"),
    "802": ("Executive direction and management", "800"),
    "803": ("Central fiscal operations", "800"),
    "804": ("General property and records management", "800"),
    "805": ("Central personnel management", "800"),
    "806": ("General purpose fiscal assistance", "800"),
    "808": ("Other general government", "800"),
    "809": ("Deductions for offsetting receipts", "800"),
    "901": ("Interest on Treasury debt securities (gross)", "900"),
    "908": ("Other interest", "900"),
    "999": ("Multiple functions", "990"),
}


def _matches(description: str, table: dict) -> list[str]:
    query = description.strip().lower()
    titles = {code: (title if isinstance(title, str) else title[0]) for code, title in table.items()}
    substring_hits = [code for code, title in titles.items() if query in title.lower() or title.lower() in query]
    if substring_hits:
        return substring_hits
    close = difflib.get_close_matches(query, [t.lower() for t in titles.values()], n=5, cutoff=0.6)
    return [code for code, title in titles.items() if title.lower() in close]


@beta_tool
def resolve_budget_function(description: str) -> str:
    """Find the budget_function/budget_subfunction code(s) matching a plain-English budget category, for get_spending_explorer_breakdown's budget_function/budget_subfunction filters, e.g. "Health", "National Defense", or "Medicare". Use this before those filters whenever the question names a category rather than already having a code from a prior call's result.

    Args:
        description: A plain-English budget function or subfunction name, e.g. "National Defense" or "Unemployment compensation".
    """
    if (over_budget := _check_tool_call_budget()) is not None:
        return over_budget

    function_hits = _matches(description, BUDGET_FUNCTIONS)
    subfunction_hits = _matches(description, BUDGET_SUBFUNCTIONS)

    if not function_hits and not subfunction_hits:
        return f"No budget function or subfunction found matching '{description}'."

    _record_tool_call(
        "resolve_budget_function",
        {"functions": function_hits, "subfunctions": subfunction_hits},
        {"description": description},
    )

    lines = [f"{code} - {BUDGET_FUNCTIONS[code]} (use as budget_function)" for code in sorted(function_hits)]
    lines += [
        f"{code} - {BUDGET_SUBFUNCTIONS[code][0]} (use as budget_subfunction, "
        f"under budget_function {BUDGET_SUBFUNCTIONS[code][1]})"
        for code in sorted(subfunction_hits)
    ]
    return _wrap_untrusted("\n".join(lines))

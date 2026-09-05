"""Typed arithmetic tools so the agent never does multi-number math in its
own prose. Pure functions - no API calls, no side effects - covering the
math shapes analyst questions actually take: totals, averages, one value's
share of a whole, a value's change over time, a same-period comparison
between two entities, and ranking several such results.

Deliberately not a generic calculate(expression: str) tool: a free-form
expression string re-opens exactly the "trust the model to get this right"
problem being closed here, and the Anthropic cookbook's
calculator_tool.ipynb example does this and explicitly calls it out as bad
practice. A small, fixed set of typed operations matches how every other
tool in this project is built (search_awards, get_spending_by_category,
etc. all take specific typed parameters, not a free-form query object).

Also deliberately not Anthropic's code_execution tool: sandbox/container
billing on top of token cost is disproportionate for arithmetic on numbers
our own tools already returned. A code_execution fallback for genuinely
open-ended computation (compound multi-step math, statistics beyond what's
covered here) is a separate, later, deliberately scoped task - see
BACKLOG.md.
"""
from __future__ import annotations

from anthropic import beta_tool

# sum_values/average/delta take an as_currency flag rather than assuming
# one - the arithmetic they do is identical either way, and the model
# already knows whether the numbers it's passing in are dollar amounts
# (it just read them from a prior tool's output), so this is a
# presentation choice, not the kind of judgment call these tools exist to
# take away from the model. rank_values sidesteps this entirely by leaving
# its values unformatted, since it's just as likely to rank percentages or
# deltas as raw dollar amounts.


@beta_tool
def sum_values(values: list[float], as_currency: bool = True) -> str:
    """Add up a list of numbers, e.g. combining several categories' or
    periods' spending amounts into one total. Use this any time you need
    the sum of two or more numbers from tool results — never add them
    yourself in prose.

    Args:
        values: The numbers to add together.
        as_currency: Whether these are dollar amounts (default True, since
            almost everything in this app is spending data) — set False
            for a non-currency quantity, e.g. a count of awards.
    """
    total = sum(values)
    return f"${total:,.2f}" if as_currency else f"{total:,.2f}"


@beta_tool
def average(values: list[float], as_currency: bool = True) -> str:
    """Compute the mean of a list of numbers, e.g. the average spending
    amount across several categories or time periods. Use this any time you
    need an average of two or more numbers from tool results — never
    average them yourself in prose.

    Args:
        values: The numbers to average. Must be non-empty.
        as_currency: Whether these are dollar amounts (default True, since
            almost everything in this app is spending data) — set False
            for a non-currency quantity, e.g. a count of awards.
    """
    if not values:
        return "Cannot compute an average of an empty list of values."
    mean = sum(values) / len(values)
    return f"${mean:,.2f}" if as_currency else f"{mean:,.2f}"


@beta_tool
def percentage_of(part: float, whole: float) -> str:
    """Compute what percentage one value is of a larger total — e.g. "what
    percent of DoD's total spending was IT contracts." Use this for ONE
    value's share of a larger total.

    Not for comparing two independent values at the same point in time
    (use ratio instead) and not for a value's change over time (use delta
    instead).

    Args:
        part: The smaller value (the share).
        whole: The larger total that `part` is a portion of.
    """
    if whole == 0:
        return "Cannot compute a percentage of a whole that is zero."
    return f"{(part / whole) * 100:.1f}%"


@beta_tool
def delta(before: float, after: float, as_currency: bool = True) -> str:
    """Compute the absolute and percentage change in the SAME thing across
    TWO POINTS IN TIME — e.g. NSF's spending in FY2023 vs. FY2024.

    Not for comparing two different entities at the same point in time
    (use ratio instead) and not for one value's share of a total (use
    percentage_of instead).

    Args:
        before: The earlier value.
        after: The later value.
        as_currency: Whether these are dollar amounts (default True, since
            almost everything in this app is spending data) — set False
            for a non-currency quantity, e.g. a count of awards. Only
            affects how the absolute change is formatted; the percentage
            change is always a percentage regardless.
    """
    absolute_change = after - before
    if before == 0:
        percent_str = "undefined (started from zero)"
    else:
        percent_str = f"{(absolute_change / before) * 100:+.1f}%"
    if absolute_change > 0:
        direction = "an increase"
    elif absolute_change < 0:
        direction = "a decrease"
    else:
        direction = "no change"
    absolute_str = f"${absolute_change:+,.2f}" if as_currency else f"{absolute_change:+,.2f}"
    return f"{absolute_str} ({percent_str}), {direction}"


@beta_tool
def ratio(a: float, a_label: str, b: float, b_label: str) -> str:
    """Compare TWO DIFFERENT entities at the SAME point in time — e.g.
    Alaska's funding vs. Alabama's funding in the same fiscal year.

    Not for the same thing changing over time (use delta instead) and not
    for one value's share of a total (use percentage_of instead).

    Args:
        a: The first value.
        a_label: A short label identifying what `a` is (e.g. "Alaska").
        b: The second value.
        b_label: A short label identifying what `b` is (e.g. "Alabama").
    """
    if b == 0:
        return f"Cannot compute a ratio: {b_label} is zero."
    return f"{a_label} is {a / b:.1f}x {b_label}"


@beta_tool
def rank_values(labeled_values: dict[str, float], descending: bool = True) -> str:
    """Sort labeled values into ranked order — e.g. ranking several
    agencies' already-computed percentage growth (from delta) to answer
    "which grew fastest." Use this after computing a delta, percentage, or
    other per-item value with the tools above, to order the results — not
    to do the underlying math itself.

    Ties are broken by the order items appear in `labeled_values` (a
    stable sort) — the first-listed of two tied items ranks higher.

    Args:
        labeled_values: A mapping from a label (e.g. an agency name) to its
            numeric value.
        descending: If True (default), rank highest value first. If False,
            rank lowest value first.
    """
    if not labeled_values:
        return "No values to rank."
    ranked = sorted(labeled_values.items(), key=lambda kv: kv[1], reverse=descending)
    lines = [f"{i}. {label}: {value:,.2f}" for i, (label, value) in enumerate(ranked, start=1)]
    return "\n".join(lines)

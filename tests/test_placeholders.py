from types import SimpleNamespace

from backend.app.agent.tools.placeholders import (
    format_with_placeholders,
    placeholder_reason,
)
from backend.app.agent.tools.spending_explorer import _format_spending_explorer_results
from backend.app.usaspending import SpendingExplorerResponse, SpendingExplorerResult


def test_placeholder_reason_recognizes_known_buckets():
    assert placeholder_reason("MULTIPLE RECIPIENTS")
    assert placeholder_reason("REDACTED DUE TO PII")
    assert placeholder_reason("CA-MULTIPLE DISTRICTS")
    assert placeholder_reason("AGGREGATE FINANCIAL ASSISTANCE")
    assert placeholder_reason("NOT SPECIFIED")
    assert placeholder_reason("Not Designated", "Q")
    assert placeholder_reason(None, None)


def test_placeholder_reason_leaves_real_entities_alone():
    assert placeholder_reason("LOCKHEED MARTIN CORPORATION", "ABC") is None
    assert placeholder_reason("Not Designated a Small Business") is None
    assert placeholder_reason("CA-36") is None
    assert placeholder_reason(None, "1234") is None


def test_format_moves_placeholders_after_real_rows():
    rows = [
        SimpleNamespace(name="MULTIPLE RECIPIENTS", amount=100.0),
        SimpleNamespace(name="ACME", amount=5.0),
    ]
    out = format_with_placeholders(rows, lambda r: (r.name, None), lambda r: f"{r.name}: ${r.amount:,.2f}")
    assert out.index("ACME") < out.index("Placeholder rows") < out.index("MULTIPLE RECIPIENTS")
    assert "not one recipient" in out


def test_explorer_formats_placeholder_separately():
    response = SpendingExplorerResponse(
        total=105.0,
        end_date="2026-06-30",
        results=[
            SpendingExplorerResult(id="1", type="office", name="AGGREGATE FINANCIAL ASSISTANCE", code="1", amount=100.0),
            SpendingExplorerResult(id="2", type="office", name="CMS OFFICE", code="2", amount=5.0),
        ],
    )
    out = _format_spending_explorer_results(response)
    assert out.index("CMS OFFICE") < out.index("Placeholder rows") < out.index("AGGREGATE FINANCIAL")

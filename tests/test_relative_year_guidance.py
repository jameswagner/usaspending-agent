from datetime import date

from backend.app.agent.response_shaping import relative_year_guidance


def test_october_is_ambiguous_and_asks_for_both_years():
    text = relative_year_guidance(date(2026, 10, 18))
    assert "FY2027" in text and "FY2026" in text
    assert "main answer" in text and "partial" in text
    assert "year='latest'" in text


def test_mid_year_labels_current_year_partial():
    text = relative_year_guidance(date(2026, 3, 1))
    assert "FY2026 is in progress" in text
    assert "also give FY2025" in text
    assert "year='latest'" not in text

from backend.app.retrieval.pipeline.ingest_cfda import build_chunk


def _row(
    number,
    title,
    popular_name="",
    objectives="",
    uses="",
    assistance_type="",
):
    return {
        "Program Number": number,
        "Program Title": title,
        "Popular Name (020)": popular_name,
        "Objectives (050)": objectives,
        "Uses and Use Restrictions (070)": uses,
        "Types of Assistance (060)": assistance_type,
    }


class TestBuildChunk:
    def test_full_row(self):
        row = _row(
            "93.778",
            "Grants to States for Medicaid",
            popular_name="(Medicaid; Title XIX)",
            objectives="To provide financial assistance to States.",
            uses="States must provide in and out-patient hospital services.",
            assistance_type="FORMULA GRANTS",
        )
        chunk = build_chunk(row)
        assert chunk.id == "CFDA_93.778"
        assert chunk.source == "CFDA Programs"
        assert chunk.term == "Grants to States for Medicaid"
        assert chunk.slug == "93.778"
        assert chunk.related_slugs == []
        assert chunk.text == (
            "93.778 - Grants to States for Medicaid\n\n"
            "Also known as: (Medicaid; Title XIX)\n\n"
            "To provide financial assistance to States.\n\n"
            "Uses: States must provide in and out-patient hospital services.\n\n"
            "Type of assistance: FORMULA GRANTS"
        )

    def test_minimal_row_no_popular_name_or_uses(self):
        row = _row("10.001", "Agricultural Research Basic and Applied Research", objectives="ARS provides solutions.")
        chunk = build_chunk(row)
        assert chunk.text == (
            "10.001 - Agricultural Research Basic and Applied Research\n\n"
            "ARS provides solutions."
        )

    def test_blank_fields_omitted(self):
        row = _row("10.001", "Some Program", popular_name="  ", objectives="Real objective.")
        chunk = build_chunk(row)
        assert "Also known as" not in chunk.text

from backend.app.retrieval.pipeline.ingest_psc import _code_str, build_chunk


class TestCodeStr:
    def test_float_code(self):
        assert _code_str(1005.0) == "1005"

    def test_string_alphanumeric_code(self):
        assert _code_str("7A20") == "7A20"

    def test_single_letter_code(self):
        assert _code_str("A") == "A"


class TestBuildChunk:
    COL_CODE, COL_NAME, COL_START, COL_END = 0, 1, 2, 3
    COL_FULL_NAME, COL_INCLUDES, COL_EXCLUDES, COL_NOTES = 4, 5, 6, 7

    def _row(self, code, name, full_name=None, includes=None, excludes=None, notes=None):
        return (code, name, None, None, full_name, includes, excludes, notes)

    def test_full_row(self):
        row = self._row(
            1005.0,
            "GUNS, THROUGH 30MM",
            full_name="Guns, through 30 mm",
            includes="Machine guns; Brushes, Machine Gun and Pistol.",
            excludes="Turrets, Aircraft.",
        )
        chunk = build_chunk(row)
        assert chunk.id == "PSC_1005"
        assert chunk.source == "PSC Codes"
        assert chunk.term == "GUNS, THROUGH 30MM"
        assert chunk.slug == "1005"
        assert chunk.related_slugs == []
        assert chunk.text == (
            "1005 - GUNS, THROUGH 30MM\n\n"
            "Guns, through 30 mm\n\n"
            "Includes: Machine guns; Brushes, Machine Gun and Pistol.\n\n"
            "Excludes: Turrets, Aircraft."
        )

    def test_name_only_row(self):
        # Group-level rows (e.g. "10 - WEAPONS") often have no full name/
        # includes/excludes, just a name and sometimes notes.
        row = self._row(10.0, "WEAPONS", notes="This group includes combat weapons.")
        chunk = build_chunk(row)
        assert chunk.text == "10 - WEAPONS\n\nThis group includes combat weapons."

    def test_full_name_identical_to_name_not_duplicated(self):
        row = self._row("A", "RESEARCH AND DEVELOPMENT", full_name="RESEARCH AND DEVELOPMENT")
        chunk = build_chunk(row)
        assert chunk.text.count("RESEARCH AND DEVELOPMENT") == 1

    def test_alphanumeric_code_preserved(self):
        row = self._row("7A20", "IT AND TELECOM - APPLICATION DEVELOPMENT SOFTWARE")
        chunk = build_chunk(row)
        assert chunk.slug == "7A20"
        assert chunk.id == "PSC_7A20"

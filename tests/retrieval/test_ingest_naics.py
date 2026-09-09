from pathlib import Path

import openpyxl

from backend.app.retrieval.pipeline.ingest_naics import (
    _code_str,
    build_chunk,
    load_descriptions,
    load_titles,
)


class TestCodeStr:
    def test_int_code(self):
        assert _code_str(111110) == "111110"

    def test_float_code(self):
        # openpyxl can hand back a float for a numeric-formatted cell.
        assert _code_str(111110.0) == "111110"

    def test_range_code_passed_through(self):
        # Combined sectors like Manufacturing (31-33) are stored as text,
        # not a number - found live when int() raised on the real file.
        assert _code_str("31-33") == "31-33"


class TestBuildChunk:
    def test_full_chunk(self):
        chunk = build_chunk("111110", "Soybean Farming", "Growing soybeans.", ["Soybean farming, field and seed production"])
        assert chunk.id == "NAICS_111110"
        assert chunk.source == "NAICS Codes"
        assert chunk.term == "Soybean Farming"
        assert chunk.slug == "111110"
        assert chunk.related_slugs == []
        assert chunk.page_start == 0
        assert chunk.page_end == 0
        assert chunk.paragraph_index == 0
        assert chunk.text == (
            "111110 - Soybean Farming\n\n"
            "Growing soybeans.\n\n"
            "Also known as: Soybean farming, field and seed production"
        )

    def test_no_description_or_synonyms(self):
        chunk = build_chunk("11", "Agriculture, Forestry, Fishing and Hunting", "", [])
        assert chunk.text == "11 - Agriculture, Forestry, Fishing and Hunting"

    def test_multiple_synonyms_joined(self):
        chunk = build_chunk("111120", "Oilseed (except Soybean) Farming", "desc", ["Canola farming", "Flaxseed farming"])
        assert chunk.text.endswith("Also known as: Canola farming; Flaxseed farming")


def _write_xlsx(path: Path, header: list, rows: list[list]) -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    for row in rows:
        ws.append(row)
    wb.save(path)


class TestLoadTitles:
    def test_skips_blank_separator_row(self, tmp_path):
        # Real file shape: header, then a blank row, then data - confirmed
        # against the actual 2-6 digit_2022_Codes.xlsx, not assumed.
        path = tmp_path / "titles.xlsx"
        _write_xlsx(
            path,
            ["Seq. No.", "2022 NAICS US Code", "2022 NAICS US Title"],
            [[None, None, None], [1, 11, "Agriculture, Forestry, Fishing and Hunting"]],
        )
        assert load_titles(str(path)) == {"11": "Agriculture, Forestry, Fishing and Hunting"}


class TestLoadDescriptions:
    def test_strips_html_tags(self, tmp_path):
        # Real wrinkle: the Manufacturing sector's (31-33) description
        # embeds an HTML table - found live, not assumed.
        path = tmp_path / "desc.xlsx"
        _write_xlsx(
            path,
            ["Code", "Title", "Description"],
            [[31, "ManufacturingT", "Examples include:<table><tr><td>Milk bottling</td></tr></table>Done."]],
        )
        result = load_descriptions(str(path))
        assert "<table>" not in result["31"]
        assert "Milk bottling" in result["31"]
        assert "Done." in result["31"]

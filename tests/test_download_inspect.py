import io
import zipfile
from unittest.mock import patch

from backend.app.agent.download_inspect import inspect_download

URL = "https://files.usaspending.gov/generated_downloads/x.zip"
HEADER = "award_id_piid,recipient_name,naics_code"
ROWS = '49100418C1011,"ACME, INC.",541990\n49100418C1068,RIVA INC,561210\n49100418C1099,OTHER INC,541512\n'


def make_zip(files: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def inspect(files, expected_columns=None, expected_rows=None):
    with patch("backend.app.agent.download_inspect._fetch_zip", return_value=make_zip(files)):
        return inspect_download(URL, expected_columns, expected_rows)


def test_reports_rows_columns_and_sample_for_a_good_file():
    result = inspect({"Contracts_PrimeTransactions_1.csv": f"{HEADER}\n{ROWS}"}, ["award_id_piid", "naics_code"], 3)
    assert result.problems == []
    (summary,) = result.files
    assert summary.rows == 3
    assert summary.columns == ["award_id_piid", "recipient_name", "naics_code"]
    assert summary.sample[0][1] == "ACME, INC."
    assert "3 rows" in result.render()


def test_empty_file_beside_a_populated_one_is_a_problem():
    result = inspect({"Contracts_PrimeTransactions_1.csv": f"{HEADER}\n{ROWS}", "Contracts_Subawards_1.csv": ""})
    assert any("Contracts_Subawards_1.csv is empty" in p for p in result.problems)
    assert "0 rows (empty)" in result.render()


def test_header_only_file_counts_as_empty():
    result = inspect({"a.csv": f"{HEADER}\n{ROWS}", "b.csv": f"{HEADER}\n"})
    assert any("b.csv is empty" in p for p in result.problems)


def test_every_file_empty_means_no_records():
    result = inspect({"a.csv": f"{HEADER}\n"})
    assert any("returned no records" in p for p in result.problems)


def test_missing_requested_column_is_reported():
    result = inspect({"a.csv": f"award_id_piid,recipient_name\n{ROWS}"}, ["award_id_piid", "naics_code"])
    assert any("naics_code" in p for p in result.problems)


def test_job_reporting_rows_but_files_empty_is_reported():
    result = inspect({"a.csv": f"{HEADER}\n"}, expected_rows=40)
    assert any("reported 40 rows" in p for p in result.problems)


def test_zip_without_csv_is_a_problem():
    result = inspect({"readme.txt": "hi"})
    assert any("no CSV files" in p for p in result.problems)


def test_fetch_failure_is_skipped_not_an_error():
    with patch("backend.app.agent.download_inspect._fetch_zip", return_value="the file could not be fetched"):
        result = inspect_download(URL)
    assert result.skipped and result.problems == []
    assert "skipped" in result.render()


def test_unreadable_zip_is_skipped():
    with patch("backend.app.agent.download_inspect._fetch_zip", return_value=b"not a zip"):
        result = inspect_download(URL)
    assert result.skipped == "the ZIP could not be read"


def test_quoted_newlines_count_as_one_row():
    result = inspect({"a.csv": f'{HEADER}\n1,"two\nlines",541990\n2,B,561210\n'})
    assert result.files[0].rows == 2

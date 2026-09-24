import os

from langsmith.utils import tracing_is_enabled


def test_tracing_is_off_during_tests():
    """Guards conftest.py: .env exports LANGSMITH_TRACING=true and the app calls
    load_dotenv() at import, so a regression here silently bills every test run
    against the account's monthly unique-trace quota."""
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert tracing_is_enabled() is False

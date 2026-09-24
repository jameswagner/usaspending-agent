import os

from langsmith.utils import tracing_is_enabled


def test_tracing_is_off_during_tests():
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert tracing_is_enabled() is False

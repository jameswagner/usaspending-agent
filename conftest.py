"""Tracing off before anything imports the app - .env turns it on, and a traced test run burns LangSmith quota."""
import os

if os.environ.get("LANGSMITH_TEST_TRACING", "").lower() != "true":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"


import pytest


@pytest.fixture(autouse=True)
def _no_download_inspection_fetch(monkeypatch):
    """Download inspection fetches the real ZIP; tests that exercise it patch _fetch_zip themselves."""
    monkeypatch.setattr(
        "backend.app.agent.download_inspect._fetch_zip", lambda url, max_bytes=0: "network disabled in tests"
    )

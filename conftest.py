"""Tracing off before anything imports the app - .env turns it on, and a traced test run burns LangSmith quota."""
import os

if os.environ.get("LANGSMITH_TEST_TRACING", "").lower() != "true":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

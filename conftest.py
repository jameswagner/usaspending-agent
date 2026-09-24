"""Session-wide test defaults, applied before anything imports the app.

.env exports LANGSMITH_TRACING=true and load_dotenv() runs at import time in
singletons.py and hybrid.py, so without this every pytest run ships a trace per
@traceable call - ~57 of them - to LangSmith and burns the account's monthly
unique-trace quota. load_dotenv() doesn't override an already-set variable, so
setting it here wins. Set LANGSMITH_TEST_TRACING=true to trace a run anyway.
"""
import os

if os.environ.get("LANGSMITH_TEST_TRACING", "").lower() != "true":
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

"""Central logging setup, so a failure that doesn't surface as an HTTP
error still leaves a record - e.g. a tool call failing server-side, or an
unhandled exception in /ask. Call once at process startup (main.py's
lifespan, and the CLI entry point).
"""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    # Same env var .env.example already documents but never wired up
    # (LOG_LEVEL=info) - reusing it rather than inventing a new one.
    level_name = os.environ.get("LOG_LEVEL", "info").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

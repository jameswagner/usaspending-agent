"""Hand-written query recipes: which endpoints answer a kind of question, how
to combine them, and their known limitations. The index goes in the system
prompt; the model loads a full recipe with get_recipe.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from anthropic import beta_tool

RECIPES_DIR = Path(__file__).parent / "recipes"
RECIPES_ENABLED = os.environ.get("RAW_API_AGENT_RECIPES", "1") == "1"


@lru_cache(maxsize=1)
def load_recipes() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in sorted(RECIPES_DIR.glob("*.md"))}


def recipe_index() -> str:
    lines = []
    for recipe_id, text in load_recipes().items():
        use_when = re.search(r"^use_when:\s*(.+)$", text, re.MULTILINE)
        not_when = re.search(r"^not_when:\s*(.+)$", text, re.MULTILINE)
        line = f"- {recipe_id}: use when {use_when.group(1) if use_when else '?'}"
        if not_when:
            line += f" (not when {not_when.group(1)})"
        lines.append(line)
    return "\n".join(lines)


@beta_tool
def get_recipe(recipe_id: str) -> str:
    """Load a query recipe: the endpoints, steps, calculations, and known
    limitations for one kind of question. Load every recipe whose use_when
    matches the question before calling those endpoints.

    Args:
        recipe_id: An id from the recipe index in the system prompt.
    """
    text = load_recipes().get(recipe_id)
    return text if text is not None else f"No recipe named {recipe_id!r}. Use an id from the recipe index."

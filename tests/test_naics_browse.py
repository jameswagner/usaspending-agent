from __future__ import annotations

from types import SimpleNamespace

from backend.app.agent.tools.naics import resolve_naics_code
from backend.app.usaspending import NaicsNode


def _client(children, seen):
    def get_naics_children(code):
        seen.append(code)
        return children
    return SimpleNamespace(get_naics_children=get_naics_children)


def test_a_sector_code_lists_its_official_children(monkeypatch):
    seen = []
    children = [NaicsNode(naics="5411", naics_description="Legal Services"), NaicsNode(naics="5415", naics_description="Computer Systems Design and Related Services")]
    monkeypatch.setattr("backend.app.agent.tools.naics._get_usaspending_client", lambda: _client(children, seen))
    out = resolve_naics_code.func(description=" 54 ")
    assert seen == ["54"]
    assert "5411 - Legal Services" in out and "5415 - Computer Systems Design" in out


def test_a_description_still_uses_the_semantic_index(monkeypatch):
    class Retriever:
        def retrieve(self, description, top_k):
            return [{"slug": "541511", "term": "Custom Computer Programming Services", "rerank_score": 9.0}]

    monkeypatch.setattr("backend.app.agent.tools.naics._get_naics_retriever", lambda: Retriever())
    monkeypatch.setattr("backend.app.agent.tools.naics._get_usaspending_client", lambda: _client([], []))
    assert "541511 - Custom Computer Programming Services" in resolve_naics_code.func(description="software")


def test_six_digit_codes_are_not_browsed(monkeypatch):
    seen = []
    monkeypatch.setattr("backend.app.agent.tools.naics._get_usaspending_client", lambda: _client([], seen))

    class Retriever:
        def retrieve(self, description, top_k):
            return []

    monkeypatch.setattr("backend.app.agent.tools.naics._get_naics_retriever", lambda: Retriever())
    resolve_naics_code.func(description="541512")
    assert seen == []


def test_browse_results_can_be_cited(monkeypatch):
    from backend.app.agent.response_shaping import build_tool_citation

    children = [NaicsNode(naics="5411", naics_description="Legal Services")]
    monkeypatch.setattr("backend.app.agent.tools.naics._get_usaspending_client", lambda: _client(children, []))
    recorded = []
    monkeypatch.setattr(
        "backend.app.agent.tools.naics._record_tool_call", lambda name, result, ctx: recorded.append((ctx, result))
    )
    resolve_naics_code.func(description="54")
    context, result = recorded[0]
    assert build_tool_citation("resolve_naics_code", context, result) is not None

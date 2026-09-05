import pytest
from fastapi.testclient import TestClient

from backend.app.agent.orchestrator import NOT_FOUND_MESSAGE, AgentResult
from backend.app.agent.response_shaping import ChartSpec, Citation, ToolCitation
from backend.app.main import app


@pytest.fixture
def client(monkeypatch):
    # warm_up() would otherwise load real ML models and connect to Chroma/
    # Whoosh at server startup (FastAPI's lifespan hook, triggered by using
    # TestClient as a context manager) - none of that exists or is needed
    # to test the HTTP layer in isolation, and it wouldn't exist in a fresh
    # CI checkout anyway (data/ is gitignored).
    monkeypatch.setattr("backend.app.main.warm_up", lambda: None)
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_agent_response(client, monkeypatch):
    fake_result = AgentResult(
        answer_text="A prime award is an agreement the federal government makes...",
        charts=[ChartSpec(chart_type="bar", title="Spending by naics", labels=["A", "B"], values=[1.0, 2.0])],
        citations=[Citation(chunk_id="Analyst's_Guide_p8", source="Analyst's Guide", page=3)],
        tool_citations=[
            ToolCitation(tool_name="lookup_agency", parameters={"name": "NSF"}, description="Agency lookup: NSF")
        ],
    )
    monkeypatch.setattr("backend.app.main.agent_ask", lambda question: fake_result)

    resp = client.post("/ask", json={"question": "What is a prime award?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer_text"] == fake_result.answer_text
    assert data["source_type"] == "agent"
    assert data["charts"] == [fake_result.charts[0].model_dump()]
    assert data["citations"] == [{"chunk_id": "Analyst's_Guide_p8", "source": "Analyst's Guide", "page": 3}]
    assert data["tool_citations"] == [
        {"tool_name": "lookup_agency", "parameters": {"name": "NSF"}, "description": "Agency lookup: NSF"}
    ]


def test_ask_not_found_source_type(client, monkeypatch):
    fake_result = AgentResult(answer_text=NOT_FOUND_MESSAGE)
    monkeypatch.setattr("backend.app.main.agent_ask", lambda question: fake_result)

    resp = client.post("/ask", json={"question": "what's the weather today"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["source_type"] == "not_found"
    assert data["answer_text"] == NOT_FOUND_MESSAGE
    assert data["charts"] == []
    assert data["citations"] == []
    assert data["tool_citations"] == []


def test_ask_passes_question_through_to_agent(client, monkeypatch):
    received = {}

    def fake_agent_ask(question):
        received["question"] = question
        return AgentResult(answer_text="ok")

    monkeypatch.setattr("backend.app.main.agent_ask", fake_agent_ask)

    client.post("/ask", json={"question": "What is a sub-award?"})

    assert received["question"] == "What is a sub-award?"


def test_ask_missing_question_field_returns_422(client):
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_ask_wrong_type_returns_422(client):
    resp = client.post("/ask", json={"question": 12345})
    assert resp.status_code == 422


def test_ui_serves_frontend(client):
    resp = client.get("/ui/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "USASpending" in resp.text

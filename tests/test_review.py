import os
import json
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    # Configure Ollama env for tests
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3:8b")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")


def test_review_endpoint_mocks_request(monkeypatch):
    client = TestClient(app)

    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    async def fake_post(url, headers=None, json=None):
        # Emulate Ollama answer with JSON content in message.content
        content = json_module.dumps(
            {
                "strengths": ["Strong Python skills", "FastAPI experience"],
                "weaknesses": ["Limited cloud experience"],
                "suggestions": ["Gain exposure to AWS/GCP", "Add metrics to achievements"],
            }
        )
        return DummyResp({"message": {"content": content}})

    # Patch httpx.AsyncClient.post for Ollama
    import httpx
    json_module = __import__("json")

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return await fake_post(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=60: DummyClient())

    payload = {
        "resume_text": "Python FastAPI developer",
        "job_text": "Seeking Python developer with FastAPI",
    }
    r = client.post("/review", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "match_score" in data
    assert isinstance(data["strengths"], list)
    assert isinstance(data["weaknesses"], list)
    assert isinstance(data["suggestions"], list)
    # debug is False by default; raw_content should be omitted
    assert "raw_content" not in data
    assert "parsing_note" not in data


def test_review_parses_json_from_mixed_content(monkeypatch):
    client = TestClient(app)

    class DummyResp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    # Simulate model returning prose + fenced JSON
    mixed = "Here is the analysis:\n```json\n{""strengths"": [""Python""], ""weaknesses"": [""Cloud""], ""suggestions"": [""Add metrics""]}\n```"

    async def fake_post(url, headers=None, json=None):
        return DummyResp({"message": {"content": mixed}})

    import httpx

    class DummyClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return await fake_post(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", lambda timeout=120: DummyClient())

    payload = {
        "resume_text": "Python FastAPI developer",
        "job_text": "Seeking Python developer with FastAPI",
        "debug": True,
    }
    r = client.post("/review", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data["strengths"], list)
    assert isinstance(data["weaknesses"], list)
    assert isinstance(data["suggestions"], list)
    # In debug mode, we expose raw_content and parsing_note
    assert "raw_content" in data
    assert "parsing_note" in data


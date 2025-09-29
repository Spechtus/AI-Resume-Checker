import io
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_extract_txt_endpoint():
    files = {"file": ("sample.txt", b"Hello AI Resume!", "text/plain")}
    r = client.post("/extract", files=files)
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "sample.txt"
    assert data["num_characters"] > 0
    assert "Hello AI Resume" in data["text"]


def test_extract_unsupported():
    files = {"file": ("image.png", b"not an image", "image/png")}
    r = client.post("/extract", files=files)
    assert r.status_code == 400
    assert "Unsupported file type" in r.json().get("detail", "")


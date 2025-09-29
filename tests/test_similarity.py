from app.services.similarity import compute_similarity
from fastapi.testclient import TestClient
from app.main import app


def test_compute_similarity_basic():
    a = "Python FastAPI developer with NLP experience"
    b = "Looking for a Python developer with FastAPI skills"
    c = "Graphic designer, Adobe Photoshop, Illustrator"

    ab = compute_similarity(a, b)
    ac = compute_similarity(a, c)
    assert ab > ac
    assert 0.0 <= ab <= 1.0
    assert 0.0 <= ac <= 1.0


def test_similarity_endpoint():
    client = TestClient(app)
    payload = {
        "resume_text": "Python FastAPI developer with NLP experience",
        "job_text": "Looking for a Python developer with FastAPI skills",
    }
    r = client.post("/similarity", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "score" in data
    assert 0.0 <= data["score"] <= 1.0


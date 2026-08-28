from fastapi.testclient import TestClient
from backend.app.main import app


client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze():
    response = client.post(
        "/analyze",
        json={
            "input": "The Earth orbits the Sun.",
            "input_type": "claim",
            "mode": "normal",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["claim"] == "The Earth orbits the Sun."
    assert data["overall_score"] == 85
    assert data["label"] == "Highly Credible"
    assert "factors" in data
    assert "evidence" in data
    assert "uncertainty" in data


def test_invalid_input_type():
    response = client.post(
        "/analyze",
        json={
            "input": "Test claim",
            "input_type": "invalid",
            "mode": "normal",
        },
    )

    assert response.status_code == 422
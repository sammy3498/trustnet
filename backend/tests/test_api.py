from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def fake_analysis_result():
    return {
        "claim": "The Earth orbits the Sun.",
        "overall_score": 98,
        "label": "Highly Credible",
        "factors": {
            "evidence_strength": {
                "score": 98,
                "justification": "Reliable evidence supports the claim.",
            },
            "factual_consistency": {
                "score": 100,
                "justification": "The claim matches the evidence.",
            },
            "source_provenance": {
                "score": 95,
                "justification": "The evidence includes a strong source.",
            },
            "bias_framing": {
                "score": 100,
                "justification": "The claim is neutrally framed.",
            },
            "incentive_conflict": {
                "score": 98,
                "justification": "No meaningful conflict was identified.",
            },
        },
        "flags": [],
        "uncertainty": {
            "gate_triggered": False,
            "level": "low",
            "reason": "Sufficient consistent evidence is available.",
        },
        "evidence_analysis": {
            "evidence_assessments": [
                {
                    "title": "NASA Evidence",
                    "url": "https://science.nasa.gov/example",
                    "stance": "supports",
                    "quality": "high",
                    "relevance": "high",
                    "reasoning": "NASA evidence supports the claim.",
                }
            ]
        },
        "evidence": [
            {
                "title": "NASA Evidence",
                "url": "https://science.nasa.gov/example",
                "source": "science.nasa.gov",
                "text": "Evidence text.",
                "quality": "high",
                "bm25_score": 4.0,
                "quality_bonus_score": 2.0,
                "retrieval_score": 6.0,
            }
        ],
        "latency_ms": 5000,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze(monkeypatch):
    monkeypatch.setattr(
        "backend.app.api.routes.analyze_claim",
        lambda claim, **kwargs: fake_analysis_result()
    )

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
    assert data["overall_score"] == 98
    assert data["label"] == "Highly Credible"
    assert data["confidence"] == "high"

    assert len(data["evidence"]) == 1
    assert data["evidence"][0]["stance"] == "supports"

    assert "factors" in data
    assert "uncertainty" in data

    assert data["latency_ms"] == 5000

def test_analyze_false_claim(monkeypatch):
    fake_result = {
        "claim": "The Sun orbits the Earth once every 24 hours.",
        "overall_score": 18,
        "label": "Not Credible",
        "factors": {
            "evidence_strength": {
                "score": 2,
                "justification": "Reliable evidence contradicts the claim.",
            },
            "factual_consistency": {
                "score": 5,
                "justification": "The claim conflicts with the evidence.",
            },
            "source_provenance": {
                "score": 95,
                "justification": "The evidence includes a strong source.",
            },
            "bias_framing": {
                "score": 90,
                "justification": "The claim is simply stated.",
            },
            "incentive_conflict": {
                "score": 95,
                "justification": "No meaningful conflict was identified.",
            },
        },
        "flags": [],
        "uncertainty": {
            "gate_triggered": False,
            "level": "low",
            "reason": "Sufficient consistent evidence is available.",
        },
        "evidence_analysis": {
            "evidence_assessments": [
                {
                    "title": "NASA Evidence",
                    "url": "https://science.nasa.gov/example",
                    "stance": "contradicts",
                    "quality": "high",
                    "relevance": "high",
                    "reasoning": "NASA evidence contradicts the claim.",
                }
            ]
        },
        "evidence": [
            {
                "title": "NASA Evidence",
                "url": "https://science.nasa.gov/example",
                "source": "science.nasa.gov",
                "text": "Evidence text.",
                "quality": "high",
                "bm25_score": 4.0,
                "quality_bonus_score": 2.0,
                "retrieval_score": 6.0,
            }
        ],
        "latency_ms": 5000,
    }

    monkeypatch.setattr(
        "backend.app.api.routes.analyze_claim",
        lambda claim, **kwargs: fake_result,
    )

    response = client.post(
        "/analyze",
        json={
            "input": "The Sun orbits the Earth once every 24 hours.",
            "input_type": "claim",
            "mode": "normal",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["overall_score"] == 18
    assert data["label"] == "Not Credible"
    assert data["confidence"] == "high"
    assert data["evidence"][0]["stance"] == "contradicts"
    assert data["factors"]["evidence_strength"]["score"] == 2
    
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


def test_text_input_not_implemented():
    response = client.post(
        "/analyze",
        json={
            "input": "Some article text",
            "input_type": "text",
            "mode": "normal",
        },
    )

    assert response.status_code == 501


def test_deep_mode_not_implemented():
    response = client.post(
        "/analyze",
        json={
            "input": "The Earth orbits the Sun.",
            "input_type": "claim",
            "mode": "deep",
        },
    )

    assert response.status_code == 501

def test_analyze_insufficient_evidence(monkeypatch):
    fake_result = {
        "claim": "This is bad.",
        "overall_score": None,
        "label": "Insufficient Evidence",
        "factors": None,
        "flags": [
            "Claim is too vague or subjective to verify reliably."
        ],
        "uncertainty": {
            "gate_triggered": True,
            "level": "high",
            "reason": "The statement is subjective.",
        },
        "claim_analysis": {
            "normalized_claim": "",
            "is_clear": False,
            "reason": "The statement is subjective.",
        },
        "evidence": [],
        "latency_ms": 2000,
    }

    monkeypatch.setattr(
        "backend.app.api.routes.analyze_claim",
        lambda claim, **kwargs: fake_result
    )

    response = client.post(
        "/analyze",
        json={
            "input": "This is bad.",
            "input_type": "claim",
            "mode": "normal",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["overall_score"] is None
    assert data["factors"] is None
    assert data["label"] == "Insufficient Evidence"
    assert data["confidence"] == "low"
    assert data["evidence"] == []
    assert data["uncertainty"]["gate_triggered"] is True
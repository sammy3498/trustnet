from backend.app.retrieval.evidence_builder import (
    build_evidence_from_snippet,
)


def test_snippet_fallback_preserves_source_quality():
    evidence = build_evidence_from_snippet(
        url="https://science.nasa.gov/example",
        title="NASA result",
        snippet="Earth travels around the Sun.",
    )

    assert evidence.source == "science.nasa.gov"
    assert evidence.text == "Earth travels around the Sun."
    assert evidence.quality == "high"
    assert evidence.evidence_origin == "snippet"


def test_medium_quality_snippet_preserves_source_quality():
    evidence = build_evidence_from_snippet(
        url="https://www.britannica.com/example",
        title="Britannica result",
        snippet="Example evidence.",
    )

    assert evidence.quality == "medium"
    assert evidence.evidence_origin == "snippet"


def test_default_evidence_origin_is_webpage():
    from backend.app.evidence.models import RetrievedEvidence

    evidence = RetrievedEvidence(
        title="Test",
        url="https://example.com",
        source="example.com",
        text="Test evidence",
        quality="low",
    )

    assert evidence.evidence_origin == "webpage"

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.evidence_analysis import (
    apply_retrieved_source_quality,
)


def test_retrieved_quality_overrides_llm_quality():
    evidence = [
        RetrievedEvidence(
            title="Britannica",
            url=(
                "https://www.britannica.com/"
                "science/example"
            ),
            source="britannica.com",
            text="Example.",
            quality="medium",
            evidence_origin="snippet",
        ),
        RetrievedEvidence(
            title="NASA",
            url="https://science.nasa.gov/example",
            source="science.nasa.gov",
            text="Example.",
            quality="high",
            evidence_origin="webpage",
        ),
    ]

    analysis = {
        "evidence_assessments": [
            {
                "title": "Britannica",
                "url": (
                    "https://www.britannica.com/"
                    "science/example"
                ),
                "quality": "low",
            },
            {
                "title": "NASA",
                "url": (
                    "https://science.nasa.gov/example"
                ),
                "quality": "low",
            },
        ]
    }

    result = apply_retrieved_source_quality(
        analysis=analysis,
        evidence=evidence,
    )

    assert (
        result["evidence_assessments"][0][
            "quality"
        ]
        == "medium"
    )

    assert (
        result["evidence_assessments"][1][
            "quality"
        ]
        == "high"
    )

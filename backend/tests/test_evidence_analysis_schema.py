import pytest
from pydantic import (
    ValidationError,
)

from backend.app.models.evidence_analysis import (
    EvidenceAnalysisResponse,
)


def base_response() -> dict:
    return {
        "claim": "Test claim",
        "evidence_assessments": [],
        "overall_evidence_direction": (
            "supports"
        ),
        "evidence_strength_score": 80,
        "summary": "Test summary",
    }


def test_missing_subclaims_rejected():
    data = base_response()

    with pytest.raises(
        ValidationError
    ):
        EvidenceAnalysisResponse.model_validate(
            data
        )


def test_empty_subclaims_rejected():
    data = base_response()

    data[
        "subclaim_assessments"
    ] = []

    with pytest.raises(
        ValidationError
    ):
        EvidenceAnalysisResponse.model_validate(
            data
        )


def test_nonempty_subclaims_accepted():
    data = base_response()

    data[
        "subclaim_assessments"
    ] = [
        {
            "subclaim": (
                "Test claim"
            ),
            "status": (
                "supports"
            ),
            "reasoning": (
                "Evidence supports it."
            ),
        }
    ]

    result = (
        EvidenceAnalysisResponse
        .model_validate(
            data
        )
    )

    assert len(
        result.subclaim_assessments
    ) == 1
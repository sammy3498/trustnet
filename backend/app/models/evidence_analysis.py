from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from backend.app.models.schemas import (
    LLMFactorBreakdown,
)


EvidenceDirection = Literal[
    "supports",
    "contradicts",
    "mixed",
    "insufficient",
]


class EvidenceAssessment(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    title: str
    url: str

    stance: Literal[
        "supports",
        "contradicts",
        "mixed",
        "neutral",
    ]

    quality: Literal[
        "high",
        "medium",
        "low",
    ]

    relevance: Literal[
        "high",
        "medium",
        "low",
    ]

    reasoning: str


class SubclaimAssessment(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    subclaim: str

    status: EvidenceDirection

    reasoning: str


class EvidenceAnalysisResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    claim: str

    evidence_assessments: list[
        EvidenceAssessment
    ]

    subclaim_assessments: list[
        SubclaimAssessment
    ] = Field(
        min_length=1
    )

    overall_evidence_direction: (
        EvidenceDirection
    )

    evidence_strength_score: int = (
        Field(
            ge=0,
            le=100,
        )
    )

    summary: str


class UnifiedAnalysisResponse(
    BaseModel
):
    claim: str

    evidence_assessments: list[
        EvidenceAssessment
    ]

    subclaim_assessments: list[
        SubclaimAssessment
    ] = Field(
        default_factory=list
    )

    overall_evidence_direction: (
        EvidenceDirection
    )

    evidence_strength_score: int = (
        Field(
            ge=0,
            le=100,
        )
    )

    evidence_summary: str

    factors: LLMFactorBreakdown

    flags: list[str]
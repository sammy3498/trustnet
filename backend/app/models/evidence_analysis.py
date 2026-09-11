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

EvidenceDirectness = Literal[
    "direct",
    "partial",
    "indirect",
    "none",
]

EvidenceCoverage = Literal[
    "complete",
    "partial",
    "none",
]

QualifierRelation = Literal[
    "matches",
    "conflicts",
    "unresolved",
    "not_applicable",
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


class EvidenceAssessmentV4(
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

    directness: (
        EvidenceDirectness
    )

    qualifier_relation: (
        QualifierRelation
    )

    reasoning: str


class SubclaimAssessmentV4(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    subclaim: str

    coverage: EvidenceCoverage

    direct_support: bool

    direct_contradiction: bool

    material_context_conflict: bool

    status: EvidenceDirection

    reasoning: str


class EvidenceAnalysisResponseV4(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    claim: str

    evidence_assessments: list[
        EvidenceAssessmentV4
    ]

    subclaim_assessments: list[
        SubclaimAssessmentV4
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
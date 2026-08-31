from typing import Literal

from pydantic import BaseModel, Field
from backend.app.models.schemas import LLMFactorBreakdown


class UnifiedAnalysisResponse(BaseModel):
    claim: str

    evidence_assessments: list[EvidenceAssessment]

    overall_evidence_direction: Literal[
        "supports",
        "contradicts",
        "mixed",
        "insufficient",
    ]

    evidence_strength_score: int = Field(
        ge=0,
        le=100,
    )

    evidence_summary: str

    factors: LLMFactorBreakdown

    flags: list[str]

class EvidenceAssessment(BaseModel):
    title: str
    url: str
    stance: Literal[
        "supports",
        "contradicts",
        "mixed",
        "neutral",
    ]
    quality: Literal["high", "medium", "low"]
    relevance: Literal["high", "medium", "low"]
    reasoning: str


class EvidenceAnalysisResponse(BaseModel):
    claim: str
    evidence_assessments: list[EvidenceAssessment]
    overall_evidence_direction: Literal[
        "supports",
        "contradicts",
        "mixed",
        "insufficient",
    ]
    evidence_strength_score: int = Field(ge=0, le=100)
    summary: str
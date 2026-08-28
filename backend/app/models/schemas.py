from typing import Literal
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    input: str
    input_type: Literal["claim", "text", "url"]
    mode: Literal["normal", "deep"] = "normal"


class FactorScore(BaseModel):
    score: int = Field(ge=0, le=100)
    justification: str


class EvidenceItem(BaseModel):
    title: str
    url: str
    source: str
    stance: Literal["supports", "contradicts", "mixed", "neutral"]
    quality: Literal["high", "medium", "low"]
    summary: str


class FactorBreakdown(BaseModel):
    evidence_strength: FactorScore
    factual_consistency: FactorScore
    source_provenance: FactorScore
    bias_framing: FactorScore
    incentive_conflict: FactorScore


class UncertaintyInfo(BaseModel):
    gate_triggered: bool
    level: Literal["low", "medium", "high"]
    reason: str


class AnalysisResponse(BaseModel):
    claim: str
    overall_score: int = Field(ge=0, le=100)

    label: Literal[
        "Highly Credible",
        "Moderately Credible",
        "Low Credibility",
        "Not Credible",
        "Insufficient Evidence",
    ]

    confidence: Literal["low", "medium", "high"]

    factors: FactorBreakdown
    uncertainty: UncertaintyInfo

    evidence: list[EvidenceItem]
    flags: list[str]

    latency_ms: int
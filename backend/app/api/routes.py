from fastapi import APIRouter

from backend.app.models.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    EvidenceItem,
    FactorBreakdown,
    FactorScore,
    UncertaintyInfo,
)
from backend.app.scoring.calculator import calculate_overall_score, get_label
from backend.app.scoring.uncertainty import evaluate_uncertainty


router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalyzeRequest):
    factors = FactorBreakdown(
        evidence_strength=FactorScore(
            score=90,
            justification="Mock evidence strongly supports the claim.",
        ),
        factual_consistency=FactorScore(
            score=85,
            justification="Mock evidence is largely consistent with the claim.",
        ),
        source_provenance=FactorScore(
            score=80,
            justification="Mock sources have strong provenance.",
        ),
        bias_framing=FactorScore(
            score=75,
            justification="Mock framing shows limited bias.",
        ),
        incentive_conflict=FactorScore(
            score=70,
            justification="No major mock conflicts were identified.",
        ),
    )

    uncertainty_gate, uncertainty_level, uncertainty_reason = evaluate_uncertainty(
        evidence_count=3,
        high_quality_count=2,
        conflicting_evidence=False,
        claim_is_clear=True,
    )

    overall_score = calculate_overall_score(
        evidence_strength=factors.evidence_strength.score,
        factual_consistency=factors.factual_consistency.score,
        source_provenance=factors.source_provenance.score,
        bias_framing=factors.bias_framing.score,
        incentive_conflict=factors.incentive_conflict.score,
    )

    label = get_label(overall_score, uncertainty_gate)

    evidence = [
        EvidenceItem(
            title="Mock Evidence Source",
            url="https://example.com/evidence",
            source="Example",
            stance="supports",
            quality="high",
            summary="This is temporary mock evidence used to test the API.",
        )
    ]

    return AnalysisResponse(
        claim=request.input,
        overall_score=overall_score,
        label=label,
        confidence="high",
        factors=factors,
        uncertainty=UncertaintyInfo(
            gate_triggered=uncertainty_gate,
            level=uncertainty_level,
            reason=uncertainty_reason,
        ),
        evidence=evidence,
        flags=[],
        latency_ms=0,
    )
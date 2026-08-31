import time
from concurrent.futures import ThreadPoolExecutor

from backend.app.evidence.conflict import detect_conflicting_evidence
from backend.app.evidence.models import RetrievedEvidence
from backend.app.llm.evidence_analysis import analyze_evidence_with_llm
from backend.app.llm.scoring import score_claim_with_llm
from backend.app.scoring.calculator import calculate_overall_score, get_label
from backend.app.scoring.uncertainty import evaluate_uncertainty


def timed_evidence_analysis(
    claim: str,
    evidence: list[RetrievedEvidence],
):
    start = time.perf_counter()

    result = analyze_evidence_with_llm(
        claim=claim,
        evidence=evidence,
    )

    elapsed_ms = round(
        (time.perf_counter() - start) * 1000
    )

    return result, elapsed_ms


def timed_credibility_scoring(
    claim: str,
    evidence: list[RetrievedEvidence],
):
    start = time.perf_counter()

    result = score_claim_with_llm(
        claim=claim,
        evidence=evidence,
    )

    elapsed_ms = round(
        (time.perf_counter() - start) * 1000
    )

    return result, elapsed_ms


def analyze_claim_with_llm(
    claim: str,
    evidence: list[RetrievedEvidence],
) -> dict:
    with ThreadPoolExecutor(max_workers=2) as executor:
        evidence_future = executor.submit(
            timed_evidence_analysis,
            claim,
            evidence,
        )

        scoring_future = executor.submit(
            timed_credibility_scoring,
            claim,
            evidence,
        )

        evidence_analysis, evidence_analysis_ms = (
            evidence_future.result()
        )

        llm_result, credibility_scoring_ms = (
            scoring_future.result()
        )

    factors = {
        "evidence_strength": {
            "score": evidence_analysis[
                "evidence_strength_score"
            ],
            "justification": evidence_analysis["summary"],
        },
        **llm_result["factors"],
    }

    overall_score = calculate_overall_score(
        evidence_strength=factors["evidence_strength"]["score"],
        factual_consistency=factors["factual_consistency"]["score"],
        source_provenance=factors["source_provenance"]["score"],
        bias_framing=factors["bias_framing"]["score"],
        incentive_conflict=factors["incentive_conflict"]["score"],
    )

    evidence_count = len(evidence)

    high_quality_sources = {
        item.source
        for item in evidence
        if item.quality == "high"
    }

    high_quality_count = len(high_quality_sources)

    conflicting_evidence = detect_conflicting_evidence(
        evidence_analysis["evidence_assessments"]
    )

    (
        uncertainty_gate,
        uncertainty_level,
        uncertainty_reason,
    ) = evaluate_uncertainty(
        evidence_count=evidence_count,
        high_quality_count=high_quality_count,
        conflicting_evidence=conflicting_evidence,
        claim_is_clear=True,
    )

    label = get_label(
        overall_score,
        uncertainty_gate=uncertainty_gate,
    )

    return {
        "llm_timings": {
            "evidence_analysis_ms": evidence_analysis_ms,
            "credibility_scoring_ms": credibility_scoring_ms,
        },
        "claim": llm_result["claim"],
        "overall_score": overall_score,
        "label": label,
        "factors": factors,
        "flags": llm_result["flags"],
        "uncertainty": {
            "gate_triggered": uncertainty_gate,
            "level": uncertainty_level,
            "reason": uncertainty_reason,
        },
        "evidence_analysis": evidence_analysis,
    }
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from fastapi import APIRouter, HTTPException

from backend.app.analysis.pipeline import analyze_claim
from backend.app.models.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    EvidenceItem,
    FactorBreakdown,
    FactorScore,
    UncertaintyInfo,
)
from backend.app.utils.config import get_mode_limits


logger = logging.getLogger(__name__)
router = APIRouter()


def derive_confidence(uncertainty: dict) -> str:
    if uncertainty["gate_triggered"]:
        return "low"

    if uncertainty["level"] == "medium":
        return "medium"

    return "high"


def build_public_evidence(result: dict) -> list[EvidenceItem]:
    retrieved_by_url = {
        item["url"].rstrip("/"): item
        for item in result.get("evidence", [])
    }
    assessments = result.get(
        "evidence_analysis", {}
    ).get("evidence_assessments", [])

    public_evidence = []
    assessed_urls = set()
    for assessment in assessments:
        url = assessment.get("url", "").rstrip("/")
        retrieved = retrieved_by_url.get(url)
        if not retrieved or url in assessed_urls:
            continue
        assessed_urls.add(url)
        public_evidence.append(
            EvidenceItem(
                title=retrieved["title"],
                url=retrieved["url"],
                source=retrieved["source"],
                stance=assessment["stance"],
                quality=retrieved["quality"],
                summary=assessment["reasoning"],
            )
        )

    # Timeout is not absence of evidence. Display any retrieved sources as
    # unassessed (neutral), never as supporting/contradicting the claim.
    for url, retrieved in retrieved_by_url.items():
        if url in assessed_urls:
            continue
        public_evidence.append(
            EvidenceItem(
                title=retrieved["title"],
                url=retrieved["url"],
                source=retrieved["source"],
                stance="neutral",
                quality=retrieved["quality"],
                summary=(
                    "Source retrieved, but the LLM did not finish assessing "
                    "its relevance or stance. No conclusion is implied."
                ),
            )
        )
    # Surface the most decisive, material and trustworthy citations first.
    # A high-authority *neutral* government page must not outrank a direct
    # counterexample from a medium-authority elections specialist.
    assessments_by_url = {
        item.get("url", "").rstrip("/"): item for item in assessments
    }
    def citation_priority(item):
        assessed = assessments_by_url.get(item.url.rstrip("/"), {})
        stance = assessed.get("stance", "neutral")
        directness = assessed.get("directness", "none")
        relevance = assessed.get("relevance", "low")
        material = (stance in {"supports", "contradicts", "mixed"}
                    and directness in {"direct", "partial"}
                    and relevance in {"high", "medium"})
        retrieved = retrieved_by_url.get(item.url.rstrip("/"), {})
        return (
            1 if material else 0,
            {"direct": 2, "partial": 1}.get(directness, 0),
            {"high": 2, "medium": 1, "low": 0}.get(relevance, 0),
            {"high": 2, "medium": 1, "low": 0}.get(item.quality, 0),
            1 if retrieved.get("evidence_origin") == "webpage" else 0,
        )
    public_evidence.sort(key=citation_priority, reverse=True)
    return public_evidence


def _hard_timeout_result(
    claim: str,
    mode: str,
    elapsed_ms: int,
) -> dict:
    reason = (
        f"{mode.title()} Mode reached the 60 second processing limit "
        "before a complete validated result was available."
    )

    return {
        "claim": claim,
        "overall_score": None,
        "label": "Insufficient Evidence",
        "factors": None,
        "flags": [reason],
        "uncertainty": {
            "gate_triggered": True,
            "level": "high",
            "reason": reason,
        },
        "evidence": [],
        "evidence_analysis": {
            "evidence_assessments": [],
            "overall_evidence_direction": "insufficient",
        },
        "timings": {},
        "llm_timings": {},
        "latency_ms": elapsed_ms,
    }


def _run_with_hard_limit(
    request: AnalyzeRequest,
) -> dict:
    limits = get_mode_limits(request.mode)
    start = time.perf_counter()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        analyze_claim,
        claim=request.input,
        mode=request.mode,
    )

    api_guard_seconds = max(
        1.0,
        limits["hard_limit_seconds"] - 0.5,
    )

    try:
        return future.result(timeout=api_guard_seconds)
    except FutureTimeoutError:
        future.cancel()
        elapsed_ms = round((time.perf_counter() - start) * 1000)
        logger.error(
            "TrustNet %s mode hit the API hard timeout after %sms.",
            request.mode,
            elapsed_ms,
        )
        return _hard_timeout_result(
            claim=request.input,
            mode=request.mode,
            elapsed_ms=elapsed_ms,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
def analyze(request: AnalyzeRequest):
    if request.input_type != "claim":
        raise HTTPException(
            status_code=501,
            detail=(
                "Only claim input is currently supported by the Phase 2 MVP."
            ),
        )

    try:
        result = _run_with_hard_limit(request)
    except Exception as exc:
        logger.exception("TrustNet credibility analysis failed.")
        raise HTTPException(
            status_code=502,
            detail="Credibility analysis failed.",
        ) from exc

    timings = result.get("timings", {})
    llm_timings = result.get("llm_timings", {})

    logger.warning(
        (
            "TrustNet timing | mode=%s | claim_preparation=%sms | "
            "retrieval=%sms | llm_analysis=%sms | evidence_analysis=%sms | "
            "credibility_scoring=%sms | total=%sms"
        ),
        request.mode,
        timings.get("claim_preparation_ms"),
        timings.get("retrieval_ms"),
        timings.get("llm_analysis_ms"),
        llm_timings.get("evidence_analysis_ms"),
        llm_timings.get("credibility_scoring_ms"),
        result.get("latency_ms"),
    )

    factors = None

    if result["factors"] is not None:
        factors = FactorBreakdown(
            evidence_strength=FactorScore(
                **result["factors"]["evidence_strength"]
            ),
            factual_consistency=FactorScore(
                **result["factors"]["factual_consistency"]
            ),
            source_provenance=FactorScore(
                **result["factors"]["source_provenance"]
            ),
            bias_framing=FactorScore(
                **result["factors"]["bias_framing"]
            ),
            incentive_conflict=FactorScore(
                **result["factors"]["incentive_conflict"]
            ),
        )

    uncertainty = UncertaintyInfo(**result["uncertainty"])

    return AnalysisResponse(
        claim=result["claim"],
        overall_score=None if uncertainty.gate_triggered else result["overall_score"],
        label=result["label"],
        confidence=derive_confidence(result["uncertainty"]),
        factors=factors,
        uncertainty=uncertainty,
        evidence=build_public_evidence(result),
        flags=result["flags"],
        latency_ms=result["latency_ms"],
    )

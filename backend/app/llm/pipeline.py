import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)

from openai import APITimeoutError

from backend.app.evidence.conflict import (
    detect_conflicting_evidence,
)
from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.evidence_analysis import (
    analyze_evidence_with_llm,
)
from backend.app.scoring.calculator import (
    calculate_overall_score,
    get_label,
)
from backend.app.scoring.calibrator import (
    build_calibrated_factors,
)
from backend.app.scoring.uncertainty import (
    evaluate_uncertainty,
)


LLM_ANALYSIS_WALL_TIMEOUT_SECONDS = (
    56.0
)
LLM_ANALYSIS_DEADLINE_RESERVE_SECONDS = (
    1.5
)
LLM_API_TIMEOUT_RESERVE_SECONDS = (
    1.0
)
MIN_LLM_ANALYSIS_WINDOW_SECONDS = (
    0.5
)


def resolve_llm_wall_timeout(
    deadline: float | None,
) -> float:
    if deadline is None:
        return (
            LLM_ANALYSIS_WALL_TIMEOUT_SECONDS
        )

    remaining = (
        deadline
        - time.perf_counter()
        - LLM_ANALYSIS_DEADLINE_RESERVE_SECONDS
    )

    return max(
        0.0,
        min(
            LLM_ANALYSIS_WALL_TIMEOUT_SECONDS,
            remaining,
        ),
    )


def resolve_llm_api_timeout(
    wall_timeout_seconds: float,
) -> float:
    return max(
        0.1,
        wall_timeout_seconds
        - LLM_API_TIMEOUT_RESERVE_SECONDS,
    )


def build_evidence_timeout_result(
    claim: str,
    evidence_analysis_ms: int,
) -> dict:
    reason = (
        "Evidence interpretation did not "
        "complete within the Normal Mode "
        "time budget."
    )

    return {
        "llm_timings": {
            "evidence_analysis_ms": (
                evidence_analysis_ms
            ),
            "credibility_scoring_ms": 0,
        },
        "claim": claim,
        "overall_score": None,
        "label": (
            "Insufficient Evidence"
        ),
        "factors": None,
        "flags": [
            reason
        ],
        "uncertainty": {
            "gate_triggered": True,
            "level": "high",
            "reason": reason,
        },
        "evidence_analysis": {
            "evidence_assessments": [],
            "subclaim_assessments": [],
            "overall_evidence_direction": (
                "insufficient"
            ),
            "summary": reason,
        },
        "analysis_degraded": True,
    }


def run_evidence_analysis(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    timeout_seconds: float | None = None,
):
    start = time.perf_counter()

    if timeout_seconds is None:
        result = (
            analyze_evidence_with_llm(
                claim=claim,
                evidence=evidence,
            )
        )

    else:
        result = (
            analyze_evidence_with_llm(
                claim=claim,
                evidence=evidence,
                timeout_seconds=(
                    timeout_seconds
                ),
            )
        )

    elapsed_ms = round(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

    return (
        result,
        elapsed_ms,
    )


def analyze_claim_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    deadline: float | None = None,
) -> dict:
    stage_start = (
        time.perf_counter()
    )

    wall_timeout_seconds = (
        resolve_llm_wall_timeout(
            deadline
        )
    )

    if (
        wall_timeout_seconds
        < MIN_LLM_ANALYSIS_WINDOW_SECONDS
    ):
        elapsed_ms = round(
            (
                time.perf_counter()
                - stage_start
            )
            * 1000
        )

        return (
            build_evidence_timeout_result(
                claim=claim,
                evidence_analysis_ms=(
                    elapsed_ms
                ),
            )
        )

    api_timeout_seconds = None

    if deadline is not None:
        api_timeout_seconds = (
            resolve_llm_api_timeout(
                wall_timeout_seconds
            )
        )

    executor = ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        run_evidence_analysis,
        claim,
        evidence,
        api_timeout_seconds,
    )

    try:
        try:
            (
                evidence_analysis,
                evidence_analysis_ms,
            ) = future.result(
                timeout=(
                    wall_timeout_seconds
                )
            )

        except (
            FuturesTimeoutError,
            APITimeoutError,
        ):
            future.cancel()

            elapsed_ms = round(
                (
                    time.perf_counter()
                    - stage_start
                )
                * 1000
            )

            return (
                build_evidence_timeout_result(
                    claim=claim,
                    evidence_analysis_ms=(
                        elapsed_ms
                    ),
                )
            )

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    credibility_assessment = {
        "bias_framing": (
            evidence_analysis[
                "bias_framing"
            ]
        ),
        "bias_framing_justification": (
            evidence_analysis[
                "bias_framing_justification"
            ]
        ),
        "incentive_conflict": (
            evidence_analysis[
                "incentive_conflict"
            ]
        ),
        "incentive_conflict_justification": (
            evidence_analysis[
                "incentive_conflict_justification"
            ]
        ),
        "material_caveats": (
            evidence_analysis[
                "material_caveats"
            ]
        ),
    }

    factors = (
        build_calibrated_factors(
            evidence_analysis=(
                evidence_analysis
            ),
            credibility_assessment=(
                credibility_assessment
            ),
            evidence=evidence,
        )
    )

    overall_score = (
        calculate_overall_score(
            evidence_strength=(
                factors[
                    "evidence_strength"
                ]["score"]
            ),
            factual_consistency=(
                factors[
                    "factual_consistency"
                ]["score"]
            ),
            source_provenance=(
                factors[
                    "source_provenance"
                ]["score"]
            ),
            bias_framing=(
                factors[
                    "bias_framing"
                ]["score"]
            ),
            incentive_conflict=(
                factors[
                    "incentive_conflict"
                ]["score"]
            ),
        )
    )

    evidence_count = len(
        evidence
    )

    high_quality_sources = {
        get_publisher_domain(
            item.source
        )
        for item in evidence
        if item.quality == "high"
    }

    medium_quality_sources = {
        get_publisher_domain(
            item.source
        )
        for item in evidence
        if item.quality == "medium"
    }

    high_quality_count = len(
        high_quality_sources
    )

    medium_quality_count = len(
        medium_quality_sources
    )

    reliable_urls = {
        item.url
        for item in evidence
        if item.quality in {
            "high",
            "medium",
        }
    }

    conflicting_evidence = (
        detect_conflicting_evidence(
            evidence_analysis[
                "evidence_assessments"
            ],
            reliable_urls=(
                reliable_urls
            ),
        )
    )

    evidence_direction = (
        evidence_analysis.get(
            "overall_evidence_direction"
        )
    )

    (
        uncertainty_gate,
        uncertainty_level,
        uncertainty_reason,
    ) = evaluate_uncertainty(
        evidence_count=(
            evidence_count
        ),
        high_quality_count=(
            high_quality_count
        ),
        medium_quality_count=(
            medium_quality_count
        ),
        conflicting_evidence=(
            conflicting_evidence
        ),
        claim_is_clear=True,
        evidence_direction=(
            evidence_direction
        ),
    )

    label = get_label(
        overall_score,
        uncertainty_gate=(
            uncertainty_gate
        ),
    )

    return {
        "llm_timings": {
            "evidence_analysis_ms": (
                evidence_analysis_ms
            ),
            "credibility_scoring_ms": 0,
        },
        "claim": claim,
        "overall_score": (
            overall_score
        ),
        "label": label,
        "factors": factors,
        "flags": (
            evidence_analysis[
                "material_caveats"
            ]
        ),
        "uncertainty": {
            "gate_triggered": (
                uncertainty_gate
            ),
            "level": (
                uncertainty_level
            ),
            "reason": (
                uncertainty_reason
            ),
        },
        "evidence_analysis": (
            evidence_analysis
        ),
        "analysis_degraded": False,
    }
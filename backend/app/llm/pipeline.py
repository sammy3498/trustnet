import time
from concurrent.futures import (
    ALL_COMPLETED,
    ThreadPoolExecutor,
    wait,
)

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
from backend.app.llm.scoring import (
    score_claim_with_llm,
)
from backend.app.scoring.calculator import (
    calculate_overall_score,
    get_label,
)
from backend.app.scoring.uncertainty import (
    evaluate_uncertainty,
)
from backend.app.utils.config import (
    get_mode_limits,
)
from backend.app.utils.time_budget import (
    AnalysisBudget,
)


QUALITY_WEIGHT = {
    "high": 1.00,
    "medium": 0.75,
    "low": 0.50,
}

PROVENANCE_SCORE = {
    "high": 100,
    "medium": 70,
    "low": 40,
}

DIRECTNESS_WEIGHT = {
    "direct": 1.00,
    "partial": 0.70,
    "indirect": 0.35,
    "none": 0.00,
}

RELEVANCE_WEIGHT = {
    "high": 1.00,
    "medium": 0.75,
    "low": 0.40,
}


def _is_timeout_exception(
    exc: BaseException,
) -> bool:
    return (
        isinstance(
            exc,
            TimeoutError,
        )
        or exc.__class__.__name__
        in {
            "APITimeoutError",
            "ReadTimeout",
            "ConnectTimeout",
        }
    )


def _assessment_strength(
    assessment: dict,
) -> float:
    quality = (
        QUALITY_WEIGHT.get(
            assessment.get(
                "quality",
                "low",
            ),
            0.50,
        )
    )

    directness = (
        DIRECTNESS_WEIGHT.get(
            assessment.get(
                "directness",
                "none",
            ),
            0.00,
        )
    )

    relevance = (
        RELEVANCE_WEIGHT.get(
            assessment.get(
                "relevance",
                "low",
            ),
            0.40,
        )
    )

    return (
        100.0
        * quality
        * directness
        * relevance
    )


def _combine_directional_strengths(
    strengths: list[float],
) -> int:
    if not strengths:
        return 0

    ranked = sorted(
        strengths,
        reverse=True,
    )

    score = ranked[0]

    for additional in ranked[1:]:
        score += (
            additional
            * 0.15
        )

    return round(
        min(
            100.0,
            score,
        )
    )


def calculate_deterministic_evidence_strength(
    evidence_analysis: dict,
) -> int:
    direction = (
        evidence_analysis.get(
            "overall_evidence_direction",
            "insufficient",
        )
    )

    assessments = (
        evidence_analysis.get(
            "evidence_assessments",
            [],
        )
    )

    support_strengths = []

    contradiction_strengths = []

    for assessment in assessments:
        stance = assessment.get(
            "stance"
        )

        strength = (
            _assessment_strength(
                assessment
            )
        )

        if stance == "supports":
            support_strengths.append(
                strength
            )

        elif stance == "contradicts":
            contradiction_strengths.append(
                strength
            )

        elif stance == "mixed":
            support_strengths.append(
                strength * 0.5
            )

            contradiction_strengths.append(
                strength * 0.5
            )

    support_score = (
        _combine_directional_strengths(
            support_strengths
        )
    )

    contradiction_score = (
        _combine_directional_strengths(
            contradiction_strengths
        )
    )

    if direction == "supports":
        score = max(
            50,
            support_score,
        )

        if contradiction_score:
            score -= round(
                contradiction_score
                * 0.35
            )

        return max(
            50,
            min(
                100,
                score,
            ),
        )

    if direction == "contradicts":
        score = (
            100
            - contradiction_score
        )

        if support_score:
            score += round(
                support_score
                * 0.35
            )

        return max(
            0,
            min(
                49,
                score,
            ),
        )

    if direction == "mixed":
        return 50

    return 50


def _is_material_assessment(
    assessment: dict,
) -> bool:
    stance = assessment.get(
        "stance"
    )

    directness = assessment.get(
        "directness"
    )

    relevance = assessment.get(
        "relevance"
    )

    if stance in {
        "supports",
        "contradicts",
        "mixed",
    }:
        return True

    if (
        relevance == "high"
        and directness
        in {
            "direct",
            "partial",
        }
    ):
        return True

    return False


def calculate_deterministic_source_provenance(
    evidence_analysis: dict,
) -> tuple[
    int,
    str,
]:
    assessments = (
        evidence_analysis.get(
            "evidence_assessments",
            [],
        )
    )

    material = [
        item
        for item in assessments
        if _is_material_assessment(
            item
        )
    ]

    if not material:
        return (
            0,
            (
                "No materially relevant "
                "evidence source was available "
                "for provenance scoring."
            ),
        )

    quality_rank = {
        "high": 3,
        "medium": 2,
        "low": 1,
    }

    publisher_quality = {}

    for index, assessment in enumerate(
        material
    ):
        url = (
            assessment.get(
                "url",
                ""
            )
            .strip()
        )

        publisher = ""

        if url:
            publisher = (
                get_publisher_domain(
                    url
                )
            )

        if publisher:
            source_key = publisher

        elif url:
            source_key = url

        else:
            title = (
                assessment.get(
                    "title",
                    ""
                )
                .strip()
                .lower()
            )

            source_key = (
                title
                or f"unknown-source-{index}"
            )

        quality = (
            assessment.get(
                "quality",
                "low",
            )
        )

        if quality not in (
            quality_rank
        ):
            quality = "low"

        previous_quality = (
            publisher_quality.get(
                source_key
            )
        )

        if (
            previous_quality is None
            or quality_rank[
                quality
            ]
            > quality_rank[
                previous_quality
            ]
        ):
            publisher_quality[
                source_key
            ] = quality

    if not publisher_quality:
        return (
            0,
            (
                "No independent materially "
                "relevant evidence publisher "
                "was available for provenance "
                "scoring."
            ),
        )

    quality_counts = {
        "high": 0,
        "medium": 0,
        "low": 0,
    }

    quality_scores = []

    for quality in (
        publisher_quality.values()
    ):
        quality_counts[
            quality
        ] += 1

        quality_scores.append(
            PROVENANCE_SCORE[
                quality
            ]
        )

    score = round(
        sum(
            quality_scores
        )
        / len(
            quality_scores
        )
    )

    total_publishers = (
        len(
            quality_scores
        )
    )

    if (
        total_publishers >= 2
        and quality_counts[
            "high"
        ] >= 2
    ):
        score = min(
            100,
            score + 5,
        )

    elif (
        total_publishers >= 3
        and quality_counts[
            "high"
        ] >= 1
        and quality_counts[
            "medium"
        ] >= 1
    ):
        score = min(
            100,
            score + 3,
        )

    material_item_count = (
        len(
            material
        )
    )

    justification = (
        "Deterministic provenance score "
        f"from {total_publishers} materially "
        "relevant independent publisher"
    )

    if total_publishers != 1:
        justification += "s"

    if (
        material_item_count
        != total_publishers
    ):
        justification += (
            f" across {material_item_count} "
            "material evidence items"
        )

    justification += (
        ": "
        f"{quality_counts['high']} high-quality, "
        f"{quality_counts['medium']} medium-quality, "
        f"{quality_counts['low']} low-quality."
    )

    return (
        score,
        justification,
    )

def timed_evidence_analysis(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    timeout_seconds: (
        float | None
    ) = None,
    mode: str = "normal",
):
    start = (
        time.perf_counter()
    )

    result = (
        analyze_evidence_with_llm(
            claim=claim,
            evidence=evidence,
            timeout_seconds=(
                timeout_seconds
            ),
            mode=mode,
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


def timed_credibility_scoring(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    timeout_seconds: (
        float | None
    ) = None,
):
    start = (
        time.perf_counter()
    )

    result = (
        score_claim_with_llm(
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


def _budget_limited_result(
    claim: str,
    evidence_analysis: (
        dict | None
    ),
    evidence_analysis_ms: (
        int | None
    ),
    credibility_scoring_ms: (
        int | None
    ),
    reason: str,
) -> dict:
    return {
        "llm_timings": {
            "evidence_analysis_ms": (
                evidence_analysis_ms
            ),
            "credibility_scoring_ms": (
                credibility_scoring_ms
            ),
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
        "evidence_analysis": (
            evidence_analysis
            or {
                "evidence_assessments": [],
                "overall_evidence_direction": (
                    "insufficient"
                ),
            }
        ),
        "budget_exhausted": True,
    }


def analyze_claim_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    budget: (
        AnalysisBudget | None
    ) = None,
    mode: str = "normal",
) -> dict:
    limits = (
        get_mode_limits(
            mode
        )
    )

    if budget is None:
        budget = AnalysisBudget(
            mode=mode,
            target_seconds=(
                limits[
                    "target_seconds"
                ]
            ),
            soft_limit_seconds=(
                limits[
                    "soft_limit_seconds"
                ]
            ),
            hard_limit_seconds=(
                limits[
                    "hard_limit_seconds"
                ]
            ),
            finalization_reserve_seconds=(
                limits[
                    "finalization_reserve_seconds"
                ]
            ),
        )

    try:
        call_timeout = (
            budget.timeout_for(
                limits[
                    "llm_call_max_seconds"
                ],
                reserve_seconds=(
                    limits[
                        "finalization_reserve_seconds"
                    ]
                ),
            )
        )

    except Exception:
        reason = (
            f"{mode.title()} Mode "
            "processing budget was "
            "exhausted before the "
            "final evidence interpretation "
            "could start."
        )

        return (
            _budget_limited_result(
                claim=claim,
                evidence_analysis=None,
                evidence_analysis_ms=None,
                credibility_scoring_ms=None,
                reason=reason,
            )
        )

    executor = (
        ThreadPoolExecutor(
            max_workers=2
        )
    )

    evidence_future = (
        executor.submit(
            timed_evidence_analysis,
            claim,
            evidence,
            call_timeout,
            mode,
        )
    )

    scoring_future = (
        executor.submit(
            timed_credibility_scoring,
            claim,
            evidence,
            call_timeout,
        )
    )

    stage_timeout = min(
        call_timeout + 0.5,
        max(
            0.25,
            (
                budget.remaining()
                - limits[
                    "finalization_reserve_seconds"
                ]
            ),
        ),
    )

    (
        done,
        not_done,
    ) = (
        wait(
            [
                evidence_future,
                scoring_future,
            ],
            timeout=(
                stage_timeout
            ),
            return_when=(
                ALL_COMPLETED
            ),
        )
    )

    for future in not_done:
        future.cancel()

    executor.shutdown(
        wait=False,
        cancel_futures=True,
    )

    evidence_analysis = None
    llm_result = None

    evidence_analysis_ms = None
    credibility_scoring_ms = None

    if (
        evidence_future
        in done
    ):
        try:
            (
                evidence_analysis,
                evidence_analysis_ms,
            ) = (
                evidence_future.result()
            )

        except Exception as exc:
            if not (
                _is_timeout_exception(
                    exc
                )
            ):
                raise

    if (
        scoring_future
        in done
    ):
        try:
            (
                llm_result,
                credibility_scoring_ms,
            ) = (
                scoring_future.result()
            )

        except Exception as exc:
            if not (
                _is_timeout_exception(
                    exc
                )
            ):
                raise

    if (
        evidence_analysis
        is None
        or llm_result
        is None
    ):
        reason = (
            f"{mode.title()} Mode "
            "processing budget was reached "
            "before all final LLM analysis "
            "completed."
        )

        return (
            _budget_limited_result(
                claim=claim,
                evidence_analysis=(
                    evidence_analysis
                ),
                evidence_analysis_ms=(
                    evidence_analysis_ms
                ),
                credibility_scoring_ms=(
                    credibility_scoring_ms
                ),
                reason=reason,
            )
        )

    deterministic_evidence_strength = (
        calculate_deterministic_evidence_strength(
            evidence_analysis
        )
    )

    (
        deterministic_source_provenance,
        provenance_justification,
    ) = (
        calculate_deterministic_source_provenance(
            evidence_analysis
        )
    )

    factors = {
        "evidence_strength": {
            "score": (
                deterministic_evidence_strength
            ),
            "justification": (
                evidence_analysis[
                    "summary"
                ]
            ),
        },
        **llm_result[
            "factors"
        ],
    }

    factors[
        "source_provenance"
    ] = {
        "score": (
            deterministic_source_provenance
        ),
        "justification": (
            provenance_justification
        ),
    }

    overall_score = (
        calculate_overall_score(
            evidence_strength=(
                factors[
                    "evidence_strength"
                ][
                    "score"
                ]
            ),
            factual_consistency=(
                factors[
                    "factual_consistency"
                ][
                    "score"
                ]
            ),
            source_provenance=(
                factors[
                    "source_provenance"
                ][
                    "score"
                ]
            ),
            bias_framing=(
                factors[
                    "bias_framing"
                ][
                    "score"
                ]
            ),
            incentive_conflict=(
                factors[
                    "incentive_conflict"
                ][
                    "score"
                ]
            ),
        )
    )

    evidence_count = (
        len(
            evidence
        )
    )

    high_quality_sources = {
        get_publisher_domain(
            item.source
        )
        for item in evidence
        if item.quality
        == "high"
    }

    medium_quality_sources = {
        get_publisher_domain(
            item.source
        )
        for item in evidence
        if item.quality
        == "medium"
    }

    high_quality_count = (
        len(
            high_quality_sources
        )
    )

    medium_quality_count = (
        len(
            medium_quality_sources
        )
    )

    reliable_urls = {
        item.url
        for item in evidence
        if item.quality
        in {
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
    ) = (
        evaluate_uncertainty(
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
    )

    label = (
        get_label(
            overall_score,
            uncertainty_gate=(
                uncertainty_gate
            ),
        )
    )

    return {
        "llm_timings": {
            "evidence_analysis_ms": (
                evidence_analysis_ms
            ),
            "credibility_scoring_ms": (
                credibility_scoring_ms
            ),
        },
        "claim": (
            llm_result[
                "claim"
            ]
        ),
        "overall_score": (
            overall_score
        ),
        "label": label,
        "factors": factors,
        "flags": (
            llm_result[
                "flags"
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
        "budget_exhausted": False,
    }
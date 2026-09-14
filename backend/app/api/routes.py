import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)

from fastapi import (
    APIRouter,
    HTTPException,
)

from backend.app.analysis.pipeline import (
    analyze_claim,
)
from backend.app.models.schemas import (
    AnalyzeRequest,
    AnalysisResponse,
    EvidenceItem,
    FactorBreakdown,
    FactorScore,
    UncertaintyInfo,
)


logger = logging.getLogger(
    __name__
)

router = APIRouter()

NORMAL_MODE_HARD_TIMEOUT_SECONDS = (
    60.0
)


def derive_confidence(
    uncertainty: dict,
) -> str:
    if uncertainty[
        "gate_triggered"
    ]:
        return "low"

    if uncertainty[
        "level"
    ] == "medium":
        return "medium"

    return "high"


def build_public_evidence(
    result: dict,
) -> list[EvidenceItem]:
    retrieved_by_url = {
        item["url"]: item
        for item in result.get(
            "evidence",
            [],
        )
    }

    assessments = (
        result.get(
            "evidence_analysis",
            {},
        ).get(
            "evidence_assessments",
            [],
        )
    )

    public_evidence = []

    for assessment in assessments:
        retrieved = (
            retrieved_by_url.get(
                assessment["url"]
            )
        )

        if not retrieved:
            continue

        public_evidence.append(
            EvidenceItem(
                title=(
                    retrieved[
                        "title"
                    ]
                ),
                url=(
                    retrieved[
                        "url"
                    ]
                ),
                source=(
                    retrieved[
                        "source"
                    ]
                ),
                stance=(
                    assessment[
                        "stance"
                    ]
                ),
                quality=(
                    retrieved[
                        "quality"
                    ]
                ),
                summary=(
                    assessment[
                        "reasoning"
                    ]
                ),
            )
        )

    return public_evidence


def run_normal_analysis(
    claim: str,
) -> dict:
    deadline = (
        time.perf_counter()
        + NORMAL_MODE_HARD_TIMEOUT_SECONDS
    )

    executor = (
        ThreadPoolExecutor(
            max_workers=1
        )
    )

    future = executor.submit(
        analyze_claim,
        claim=claim,
        search_results=5,
        top_k=3,
        deadline=deadline,
    )

    try:
        return future.result(
            timeout=(
                NORMAL_MODE_HARD_TIMEOUT_SECONDS
            )
        )

    except FuturesTimeoutError as exc:
        future.cancel()

        logger.error(
            "TrustNet Normal Mode exceeded "
            "%.1f-second hard timeout.",
            NORMAL_MODE_HARD_TIMEOUT_SECONDS,
        )

        raise HTTPException(
            status_code=504,
            detail=(
                "TrustNet analysis exceeded "
                "the 60-second Normal Mode "
                "time limit. Please try again."
            ),
        ) from exc

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
def analyze(
    request: AnalyzeRequest,
):
    if (
        request.input_type
        != "claim"
    ):
        raise HTTPException(
            status_code=501,
            detail=(
                "Only claim input is currently "
                "supported."
            ),
        )

    if request.mode != "normal":
        raise HTTPException(
            status_code=501,
            detail=(
                "Deep Mode is not implemented yet."
            ),
        )

    try:
        result = (
            run_normal_analysis(
                request.input
            )
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "TrustNet credibility analysis failed."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "Credibility analysis failed."
            ),
        ) from exc

    timings = result.get(
        "timings",
        {},
    )

    llm_timings = result.get(
        "llm_timings",
        {},
    )

    logger.warning(
        (
            "TrustNet timing | "
            "claim_preparation=%sms | "
            "retrieval=%sms | "
            "llm_analysis=%sms | "
            "evidence_analysis=%sms | "
            "credibility_scoring=%sms | "
            "total=%sms"
        ),
        timings.get(
            "claim_preparation_ms"
        ),
        timings.get(
            "retrieval_ms"
        ),
        timings.get(
            "llm_analysis_ms"
        ),
        llm_timings.get(
            "evidence_analysis_ms"
        ),
        llm_timings.get(
            "credibility_scoring_ms"
        ),
        result.get(
            "latency_ms"
        ),
    )

    factors = None

    if result[
        "factors"
    ] is not None:
        factors = FactorBreakdown(
            evidence_strength=(
                FactorScore(
                    **result[
                        "factors"
                    ][
                        "evidence_strength"
                    ]
                )
            ),
            factual_consistency=(
                FactorScore(
                    **result[
                        "factors"
                    ][
                        "factual_consistency"
                    ]
                )
            ),
            source_provenance=(
                FactorScore(
                    **result[
                        "factors"
                    ][
                        "source_provenance"
                    ]
                )
            ),
            bias_framing=(
                FactorScore(
                    **result[
                        "factors"
                    ][
                        "bias_framing"
                    ]
                )
            ),
            incentive_conflict=(
                FactorScore(
                    **result[
                        "factors"
                    ][
                        "incentive_conflict"
                    ]
                )
            ),
        )

    uncertainty = (
        UncertaintyInfo(
            **result[
                "uncertainty"
            ]
        )
    )

    return AnalysisResponse(
        claim=result[
            "claim"
        ],
        overall_score=result[
            "overall_score"
        ],
        label=result[
            "label"
        ],
        confidence=(
            derive_confidence(
                result[
                    "uncertainty"
                ]
            )
        ),
        factors=factors,
        uncertainty=uncertainty,
        evidence=(
            build_public_evidence(
                result
            )
        ),
        flags=result[
            "flags"
        ],
        latency_ms=result[
            "latency_ms"
        ],
    )
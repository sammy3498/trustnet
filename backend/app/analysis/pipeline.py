import time

from backend.app.llm.claim_preparation import (
    prepare_claim,
)
from backend.app.llm.pipeline import (
    analyze_claim_with_llm,
)
from backend.app.retrieval.search_pipeline import (
    search_and_retrieve,
)


def analyze_claim(
    claim: str,
    search_results: int = 10,
    top_k: int = 5,
    context=None,
    retrieval_end_date: (
        str | None
    ) = None,
) -> dict:
    start_time = (
        time.perf_counter()
    )

    t0 = start_time

    claim_preparation = (
        prepare_claim(
            claim,
            context=context,
        )
    )

    t1 = time.perf_counter()

    claim_analysis = {
        "normalized_claim": (
            claim_preparation[
                "normalized_claim"
            ]
        ),
        "is_clear": (
            claim_preparation[
                "is_clear"
            ]
        ),
        "reason": (
            claim_preparation[
                "reason"
            ]
        ),
    }

    search_query = (
        claim_preparation.get(
            "search_query",
            "",
        )
    )

    verification_queries = (
        claim_preparation.get(
            "verification_queries",
            [],
        )
    )

    normalized_claim = (
        claim_preparation[
            "normalized_claim"
        ]
    )

    evidence = (
        search_and_retrieve(
            query=normalized_claim,
            alternate_query=(
                search_query
            ),
            verification_queries=(
                verification_queries
            ),
            search_results=(
                search_results
            ),
            top_k=top_k,
            end_date=(
                retrieval_end_date
            ),
        )
    )

    t2 = time.perf_counter()

    if not evidence:
        if claim_preparation[
            "is_clear"
        ]:
            reason = (
                "No usable evidence "
                "could be retrieved."
            )

            flags = [
                reason
            ]

        else:
            reason = (
                claim_preparation[
                    "reason"
                ]
            )

            flags = [
                "Claim is too vague or "
                "subjective to verify reliably.",
                "No usable evidence could "
                "be retrieved.",
            ]

        return {
            "claim": (
                normalized_claim
            ),
            "overall_score": None,
            "label": (
                "Insufficient Evidence"
            ),
            "factors": None,
            "flags": flags,
            "uncertainty": {
                "gate_triggered": True,
                "level": "high",
                "reason": reason,
            },
            "claim_analysis": (
                claim_analysis
            ),
            "search_query": (
                search_query
            ),
            "verification_queries": (
                verification_queries
            ),
            "retrieval_end_date": (
                retrieval_end_date
            ),
            "evidence": [],
            "timings": {
                "claim_preparation_ms": (
                    round(
                        (
                            t1 - t0
                        )
                        * 1000
                    )
                ),
                "retrieval_ms": (
                    round(
                        (
                            t2 - t1
                        )
                        * 1000
                    )
                ),
                "llm_analysis_ms": 0,
            },
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            ),
        }

    if not claim_preparation[
        "is_clear"
    ]:
        return {
            "claim": (
                normalized_claim
            ),
            "overall_score": None,
            "label": (
                "Insufficient Evidence"
            ),
            "factors": None,
            "flags": [
                "Claim is too vague or "
                "subjective to verify reliably.",
                "Evidence was retrieved for "
                "context, but the clarity "
                "uncertainty gate remains active.",
            ],
            "uncertainty": {
                "gate_triggered": True,
                "level": "high",
                "reason": (
                    claim_preparation[
                        "reason"
                    ]
                ),
            },
            "claim_analysis": (
                claim_analysis
            ),
            "search_query": (
                search_query
            ),
            "verification_queries": (
                verification_queries
            ),
            "retrieval_end_date": (
                retrieval_end_date
            ),
            "evidence": [
                item.model_dump()
                for item in evidence
            ],
            "timings": {
                "claim_preparation_ms": (
                    round(
                        (
                            t1 - t0
                        )
                        * 1000
                    )
                ),
                "retrieval_ms": (
                    round(
                        (
                            t2 - t1
                        )
                        * 1000
                    )
                ),
                "llm_analysis_ms": 0,
            },
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            ),
        }

    result = (
        analyze_claim_with_llm(
            claim=normalized_claim,
            evidence=evidence,
        )
    )

    t3 = time.perf_counter()

    result[
        "claim_analysis"
    ] = claim_analysis

    result["evidence"] = [
        item.model_dump()
        for item in evidence
    ]

    result[
        "search_query"
    ] = search_query

    result[
        "verification_queries"
    ] = verification_queries

    result[
        "retrieval_end_date"
    ] = retrieval_end_date

    result["timings"] = {
        "claim_preparation_ms": (
            round(
                (
                    t1 - t0
                )
                * 1000
            )
        ),
        "retrieval_ms": (
            round(
                (
                    t2 - t1
                )
                * 1000
            )
        ),
        "llm_analysis_ms": (
            round(
                (
                    t3 - t2
                )
                * 1000
            )
        ),
    }

    result[
        "latency_ms"
    ] = round(
        (
            time.perf_counter()
            - start_time
        )
        * 1000
    )

    return result
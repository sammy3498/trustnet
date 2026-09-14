import time

from backend.app.analysis.cache import (
    CACHE_TTL_SECONDS,
    build_analysis_cache_key,
    get_cached_analysis,
    set_cached_analysis,
)
from backend.app.evidence.compound_fast_path import (
    try_compound_fast_evidence_analysis,
)
from backend.app.evidence.fast_path import (
    is_simple_atomic_claim,
    try_fast_evidence_analysis,
)
from backend.app.llm.claim_preparation import (
    prepare_claim,
)
from backend.app.llm.pipeline import (
    analyze_claim_with_llm,
    score_evidence_analysis,
)
from backend.app.retrieval.authority_recovery import (
    recover_authority_evidence,
)
from backend.app.retrieval.search_pipeline import (
    search_and_retrieve,
)


def append_unique_evidence(
    evidence,
    additions,
):
    existing_urls = {
        item.url
        for item in evidence
    }

    for item in additions:
        if item.url in existing_urls:
            continue

        evidence.append(
            item
        )

        existing_urls.add(
            item.url
        )


def build_cache_hit_result(
    cached_result: dict,
    start_time: float,
) -> dict:
    cached_result["cache"] = {
        "hit": True,
        "ttl_seconds": CACHE_TTL_SECONDS,
    }

    cached_result["timings"] = {
        "claim_preparation_ms": 0,
        "retrieval_ms": 0,
        "authority_recovery_ms": 0,
        "fast_path_probe_ms": 0,
        "llm_analysis_ms": 0,
    }

    cached_result["latency_ms"] = round(
        (
            time.perf_counter()
            - start_time
        )
        * 1000
    )

    return cached_result


def should_cache_result(
    result: dict,
    claim_preparation: dict,
) -> bool:
    if claim_preparation.get(
        "timed_out",
        False,
    ):
        return False

    if result.get(
        "analysis_degraded",
        False,
    ):
        return False

    if result.get(
        "overall_score"
    ) is None:
        return False

    if result.get(
        "factors"
    ) is None:
        return False

    return True


def analyze_claim(
    claim: str,
    search_results: int = 10,
    top_k: int = 5,
    context=None,
    retrieval_end_date: (
        str | None
    ) = None,
    deadline: float | None = None,
) -> dict:
    start_time = (
        time.perf_counter()
    )

    cache_key = (
        build_analysis_cache_key(
            claim=claim,
            search_results=(
                search_results
            ),
            top_k=top_k,
            context=context,
            retrieval_end_date=(
                retrieval_end_date
            ),
        )
    )

    cached_result = (
        get_cached_analysis(
            cache_key
        )
    )

    if cached_result is not None:
        return (
            build_cache_hit_result(
                cached_result=(
                    cached_result
                ),
                start_time=(
                    start_time
                ),
            )
        )

    t0 = start_time

    claim_preparation = (
        prepare_claim(
            claim,
            context=context,
        )
    )

    t1 = (
        time.perf_counter()
    )

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

    t2 = (
        time.perf_counter()
    )

    initial_retrieval_ms = round(
        (
            t2 - t1
        )
        * 1000
    )

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
                (
                    "Claim is too vague or "
                    "subjective to verify reliably."
                ),
                (
                    "No usable evidence could "
                    "be retrieved."
                ),
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
                "claim_preparation_ms": round(
                    (
                        t1 - t0
                    )
                    * 1000
                ),
                "retrieval_ms": (
                    initial_retrieval_ms
                ),
                "authority_recovery_ms": 0,
                "fast_path_probe_ms": 0,
                "llm_analysis_ms": 0,
            },
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            ),
            "cache": {
                "hit": False,
                "ttl_seconds": (
                    CACHE_TTL_SECONDS
                ),
            },
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
                (
                    "Claim is too vague or "
                    "subjective to verify reliably."
                ),
                (
                    "Evidence was retrieved for "
                    "context, but the clarity "
                    "uncertainty gate remains active."
                ),
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
                "claim_preparation_ms": round(
                    (
                        t1 - t0
                    )
                    * 1000
                ),
                "retrieval_ms": (
                    initial_retrieval_ms
                ),
                "authority_recovery_ms": 0,
                "fast_path_probe_ms": 0,
                "llm_analysis_ms": 0,
            },
            "latency_ms": round(
                (
                    time.perf_counter()
                    - start_time
                )
                * 1000
            ),
            "cache": {
                "hit": False,
                "ttl_seconds": (
                    CACHE_TTL_SECONDS
                ),
            },
        }

    fast_path_probe_ms = 0

    authority_recovery_ms = 0

    compound_analysis = None

    compound_fast_eligible = (
        len(
            verification_queries
        )
        >= 2
        and all(
            is_simple_atomic_claim(
                item
            )
            for item
            in verification_queries
        )
    )

    if compound_fast_eligible:
        compound_result = (
            try_compound_fast_evidence_analysis(
                claim=normalized_claim,
                subclaims=(
                    verification_queries
                ),
                evidence=evidence,
                end_date=(
                    retrieval_end_date
                ),
                deadline=deadline,
            )
        )

        fast_path_probe_ms += int(
            compound_result[
                "fast_path_ms"
            ]
        )

        authority_recovery_ms += int(
            compound_result[
                "authority_recovery_ms"
            ]
        )

        append_unique_evidence(
            evidence=evidence,
            additions=(
                compound_result[
                    "recovered_evidence"
                ]
            ),
        )

        compound_analysis = (
            compound_result[
                "analysis"
            ]
        )

    elif is_simple_atomic_claim(
        normalized_claim
    ):
        probe_start = (
            time.perf_counter()
        )

        initial_fast_analysis = (
            try_fast_evidence_analysis(
                claim=normalized_claim,
                evidence=evidence,
            )
        )

        fast_path_probe_ms = round(
            (
                time.perf_counter()
                - probe_start
            )
            * 1000
        )

        if (
            initial_fast_analysis
            is None
        ):
            recovery_start = (
                time.perf_counter()
            )

            recovered_evidence = (
                recover_authority_evidence(
                    claim=normalized_claim,
                    existing_evidence=(
                        evidence
                    ),
                    end_date=(
                        retrieval_end_date
                    ),
                    deadline=deadline,
                )
            )

            authority_recovery_ms = round(
                (
                    time.perf_counter()
                    - recovery_start
                )
                * 1000
            )

            append_unique_evidence(
                evidence=evidence,
                additions=(
                    recovered_evidence
                ),
            )

    analysis_start = (
        time.perf_counter()
    )

    if compound_analysis is not None:
        result = (
            score_evidence_analysis(
                claim=normalized_claim,
                evidence=evidence,
                evidence_analysis=(
                    compound_analysis
                ),
                evidence_analysis_ms=(
                    fast_path_probe_ms
                ),
                analysis_path=(
                    "local_nli_compound"
                ),
                fast_path_ms=(
                    fast_path_probe_ms
                ),
                conflicting_evidence_override=(
                    False
                ),
            )
        )

    elif deadline is None:
        result = (
            analyze_claim_with_llm(
                claim=normalized_claim,
                evidence=evidence,
            )
        )

    else:
        result = (
            analyze_claim_with_llm(
                claim=normalized_claim,
                evidence=evidence,
                deadline=deadline,
            )
        )

    t3 = (
        time.perf_counter()
    )

    result[
        "claim_analysis"
    ] = claim_analysis

    result[
        "evidence"
    ] = [
        item.model_dump()
        for item in evidence
    ]

    result[
        "search_query"
    ] = search_query

    result[
        "verification_queries"
    ] = (
        verification_queries
    )

    result[
        "retrieval_end_date"
    ] = retrieval_end_date

    actual_analysis_ms = round(
        (
            t3
            - analysis_start
        )
        * 1000
    )

    result[
        "timings"
    ] = {
        "claim_preparation_ms": round(
            (
                t1 - t0
            )
            * 1000
        ),
        "retrieval_ms": (
            initial_retrieval_ms
            + authority_recovery_ms
        ),
        "authority_recovery_ms": (
            authority_recovery_ms
        ),
        "fast_path_probe_ms": (
            fast_path_probe_ms
        ),
        "llm_analysis_ms": (
            fast_path_probe_ms
            + actual_analysis_ms
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

    result[
        "cache"
    ] = {
        "hit": False,
        "ttl_seconds": (
            CACHE_TTL_SECONDS
        ),
    }

    if should_cache_result(
        result=result,
        claim_preparation=(
            claim_preparation
        ),
    ):
        set_cached_analysis(
            key=cache_key,
            result=result,
        )

    return result
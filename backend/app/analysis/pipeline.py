import time

from backend.app.llm.claim_preparation import prepare_claim
from backend.app.llm.pipeline import analyze_claim_with_llm
from backend.app.retrieval.search_pipeline import search_and_retrieve
from backend.app.utils.config import get_mode_limits
from backend.app.utils.time_budget import AnalysisBudget


DEEP_REFINEMENT_SUFFIXES = (
    "primary source official report data",
    "independent corroboration original evidence",
)


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


def _make_budget(
    mode: str,
    start_time: float,
) -> AnalysisBudget:
    limits = get_mode_limits(
        mode
    )

    return AnalysisBudget(
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
        started_at=start_time,
    )


def _prepare_normal_claim(
    claim: str,
    context=None,
) -> dict:
    normalized_claim = (
        claim.strip()
    )

    if not normalized_claim:
        return {
            "normalized_claim": "",
            "is_clear": False,
            "reason": (
                "The claim is empty."
            ),
            "search_query": "",
            "verification_queries": [],
        }

    if context:
        search_query = (
            f"{normalized_claim} "
            f"{str(context).strip()}"
        ).strip()
    else:
        search_query = (
            normalized_claim
        )

    return {
        "normalized_claim": (
            normalized_claim
        ),
        "is_clear": True,
        "reason": (
            "Direct factual claim prepared "
            "without LLM preprocessing in "
            "Normal Mode."
        ),
        "search_query": (
            search_query
        ),
        "verification_queries": [
            normalized_claim
        ],
    }


def _insufficient_result(
    claim: str,
    reason: str,
    flags: list[str],
    claim_analysis: dict,
    search_query: str,
    verification_queries: list[str],
    retrieval_end_date: (
        str | None
    ),
    evidence: list,
    start_time: float,
    t1: float | None = None,
    t2: float | None = None,
) -> dict:
    now = time.perf_counter()

    claim_ms = (
        0
        if t1 is None
        else round(
            (
                t1
                - start_time
            )
            * 1000
        )
    )

    retrieval_ms = 0

    if (
        t1 is not None
        and t2 is not None
    ):
        retrieval_ms = round(
            (
                t2
                - t1
            )
            * 1000
        )

    return {
        "claim": claim,
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
        "evidence": [
            (
                item.model_dump()
                if hasattr(
                    item,
                    "model_dump",
                )
                else item
            )
            for item in evidence
        ],
        "evidence_analysis": {
            "evidence_assessments": [],
            "overall_evidence_direction": (
                "insufficient"
            ),
        },
        "timings": {
            "claim_preparation_ms": (
                claim_ms
            ),
            "retrieval_ms": (
                retrieval_ms
            ),
            "llm_analysis_ms": 0,
        },
        "llm_timings": {
            "evidence_analysis_ms": (
                None
            ),
            "credibility_scoring_ms": (
                None
            ),
        },
        "latency_ms": round(
            (
                now
                - start_time
            )
            * 1000
        ),
    }


def _merge_evidence(
    current: list,
    additional: list,
    max_items: int,
) -> list:
    merged = []
    seen = set()

    for item in (
        current
        + additional
    ):
        if item.url in seen:
            continue

        seen.add(
            item.url
        )

        merged.append(
            item
        )

        if len(
            merged
        ) >= max_items:
            break

    return merged


def _deep_followup_queries(
    normalized_claim: str,
    verification_queries: (
        list[str]
    ),
    suffix: str,
) -> tuple[
    str,
    list[str],
]:
    bases = (
        verification_queries[
            :2
        ]
        or [
            normalized_claim
        ]
    )

    followups = [
        f"{base} {suffix}"
        for base in bases
    ]

    alternate = (
        f"{normalized_claim} "
        f"{suffix}"
    )

    return (
        alternate,
        followups,
    )


def analyze_claim(
    claim: str,
    search_results: (
        int | None
    ) = None,
    top_k: int | None = None,
    context=None,
    retrieval_end_date: (
        str | None
    ) = None,
    mode: str = "normal",
) -> dict:
    start_time = (
        time.perf_counter()
    )

    limits = (
        get_mode_limits(
            mode
        )
    )

    budget = (
        _make_budget(
            mode,
            start_time,
        )
    )

    if search_results is None:
        search_results = (
            limits[
                "search_results"
            ]
        )

    if top_k is None:
        top_k = (
            limits[
                "top_k"
            ]
        )

    if mode == "normal":
        claim_preparation = (
            _prepare_normal_claim(
                claim,
                context=context,
            )
        )

    else:
        try:
            preparation_timeout = (
                budget.timeout_for(
                    limits[
                        "claim_preparation_max_seconds"
                    ],
                    reserve_seconds=(
                        limits[
                            "finalization_reserve_seconds"
                        ]
                        + limits[
                            "llm_call_max_seconds"
                        ]
                        + 2.0
                    ),
                )
            )

            claim_preparation = (
                prepare_claim(
                    claim,
                    context=context,
                    timeout_seconds=(
                        preparation_timeout
                    ),
                )
            )

        except Exception as exc:
            if not (
                _is_timeout_exception(
                    exc
                )
            ):
                raise

            claim_preparation = (
                _prepare_normal_claim(
                    claim,
                    context=context,
                )
            )

            claim_preparation[
                "reason"
            ] = (
                "Deep Mode LLM claim "
                "preparation timed out; "
                "deterministic claim "
                "preparation was used "
                "instead."
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

    if not normalized_claim:
        reason = (
            claim_preparation[
                "reason"
            ]
        )

        return (
            _insufficient_result(
                claim=claim,
                reason=reason,
                flags=[
                    reason
                ],
                claim_analysis=(
                    claim_analysis
                ),
                search_query=(
                    search_query
                ),
                verification_queries=(
                    verification_queries
                ),
                retrieval_end_date=(
                    retrieval_end_date
                ),
                evidence=[],
                start_time=(
                    start_time
                ),
                t1=t1,
            )
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
            budget=budget,
            mode=mode,
        )
    )

    if (
        mode == "deep"
        and evidence
    ):
        max_deep_evidence = (
            max(
                top_k + 2,
                6,
            )
        )

        for suffix in (
            DEEP_REFINEMENT_SUFFIXES
        ):
            if (
                budget.elapsed()
                >= 34.0
            ):
                break

            if not (
                budget.can_start(
                    expected_seconds=8.0,
                    reserve_seconds=(
                        limits[
                            "llm_call_max_seconds"
                        ]
                        + limits[
                            "finalization_reserve_seconds"
                        ]
                    ),
                )
            ):
                break

            (
                alternate,
                followups,
            ) = (
                _deep_followup_queries(
                    normalized_claim=(
                        normalized_claim
                    ),
                    verification_queries=(
                        verification_queries
                    ),
                    suffix=suffix,
                )
            )

            additional = (
                search_and_retrieve(
                    query=(
                        normalized_claim
                    ),
                    alternate_query=(
                        alternate
                    ),
                    verification_queries=(
                        followups
                    ),
                    search_results=(
                        search_results
                    ),
                    top_k=(
                        min(
                            3,
                            top_k,
                        )
                    ),
                    end_date=(
                        retrieval_end_date
                    ),
                    budget=budget,
                    mode=mode,
                )
            )

            evidence = (
                _merge_evidence(
                    evidence,
                    additional,
                    max_items=(
                        max_deep_evidence
                    ),
                )
            )

    t2 = time.perf_counter()

    if not evidence:
        if (
            claim_preparation[
                "is_clear"
            ]
        ):
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
                    "Claim is too vague "
                    "or subjective to "
                    "verify reliably."
                ),
                (
                    "No usable evidence "
                    "could be retrieved."
                ),
            ]

        return (
            _insufficient_result(
                claim=(
                    normalized_claim
                ),
                reason=reason,
                flags=flags,
                claim_analysis=(
                    claim_analysis
                ),
                search_query=(
                    search_query
                ),
                verification_queries=(
                    verification_queries
                ),
                retrieval_end_date=(
                    retrieval_end_date
                ),
                evidence=[],
                start_time=(
                    start_time
                ),
                t1=t1,
                t2=t2,
            )
        )

    if not (
        claim_preparation[
            "is_clear"
        ]
    ):
        reason = (
            claim_preparation[
                "reason"
            ]
        )

        return (
            _insufficient_result(
                claim=(
                    normalized_claim
                ),
                reason=reason,
                flags=[
                    (
                        "Claim is too vague "
                        "or subjective to "
                        "verify reliably."
                    ),
                    (
                        "Evidence was retrieved "
                        "for context, but the "
                        "clarity uncertainty "
                        "gate remains active."
                    ),
                ],
                claim_analysis=(
                    claim_analysis
                ),
                search_query=(
                    search_query
                ),
                verification_queries=(
                    verification_queries
                ),
                retrieval_end_date=(
                    retrieval_end_date
                ),
                evidence=evidence,
                start_time=(
                    start_time
                ),
                t1=t1,
                t2=t2,
            )
        )

    if (
        budget.expired(
            reserve_seconds=(
                limits[
                    "finalization_reserve_seconds"
                ]
                + 0.5
            )
        )
    ):
        reason = (
            f"{mode.title()} Mode "
            "processing budget was "
            "reached before final "
            "evidence interpretation "
            "could begin."
        )

        return (
            _insufficient_result(
                claim=(
                    normalized_claim
                ),
                reason=reason,
                flags=[
                    reason
                ],
                claim_analysis=(
                    claim_analysis
                ),
                search_query=(
                    search_query
                ),
                verification_queries=(
                    verification_queries
                ),
                retrieval_end_date=(
                    retrieval_end_date
                ),
                evidence=evidence,
                start_time=(
                    start_time
                ),
                t1=t1,
                t2=t2,
            )
        )

    result = (
        analyze_claim_with_llm(
            claim=(
                normalized_claim
            ),
            evidence=evidence,
            budget=budget,
            mode=mode,
        )
    )

    t3 = time.perf_counter()

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
    ] = verification_queries

    result[
        "retrieval_end_date"
    ] = retrieval_end_date

    result[
        "mode"
    ] = mode

    result[
        "time_budget"
    ] = {
        "target_seconds": (
            limits[
                "target_seconds"
            ]
        ),
        "soft_limit_seconds": (
            limits[
                "soft_limit_seconds"
            ]
        ),
        "hard_limit_seconds": (
            limits[
                "hard_limit_seconds"
            ]
        ),
    }

    result[
        "timings"
    ] = {
        "claim_preparation_ms": (
            round(
                (
                    t1
                    - start_time
                )
                * 1000
            )
        ),
        "retrieval_ms": (
            round(
                (
                    t2
                    - t1
                )
                * 1000
            )
        ),
        "llm_analysis_ms": (
            round(
                (
                    t3
                    - t2
                )
                * 1000
            )
        ),
    }

    result[
        "latency_ms"
    ] = round(
        (
            t3
            - start_time
        )
        * 1000
    )

    if (
        mode == "normal"
        and result[
            "latency_ms"
        ]
        > round(
            limits[
                "target_seconds"
            ]
            * 1000
        )
    ):
        result.setdefault(
            "flags",
            [],
        ).append(
            (
                "Normal Mode exceeded "
                "the preferred ~20 second "
                "target but remained "
                "within the processing "
                "limit."
            )
        )

    return result
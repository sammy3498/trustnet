import os
import re
import time

from backend.app.llm.claim_preparation import prepare_claim
from backend.app.llm.pipeline import analyze_claim_with_llm
from backend.app.retrieval.search_pipeline import search_and_retrieve
from backend.app.retrieval.verification_plan import plan_verification_queries
from backend.app.retrieval.coverage_recovery import recovery_query
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
    mode: str = "normal",
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

    # Bare size comparisons between geographic entities can change truth
    # value between administrative area, metro population and city proper.
    # Don't silently select the interpretation most convenient to a source.
    if re.search(r"\b(?:larger|smaller|bigger)\s+than\b", normalized_claim,re.I) and not re.search(
        r"\b(population|area|land|square|km2|km²|mi2|mile|kilomet|diameter|radius|mass|weight|volume|economy|gdp|metro|municipal|city proper)\b",normalized_claim,re.I
    ):
        return {
            "normalized_claim":normalized_claim,"is_clear":False,
            "reason":"The comparison is underspecified: define what 'larger/smaller' measures (for example area, population, mass or another explicit metric) and the boundaries/timeframe where relevant.",
            "search_query":normalized_claim,"verification_queries":[],
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

    max_queries = get_mode_limits(mode)["max_queries"]
    search_plan = plan_verification_queries(normalized_claim, max_queries=max_queries)
    # Historical benchmark metadata is not a source of truth; its date merely
    # helps narrow search results. The claim text and literal query remain.
    date_match = re.search(r"(?im)^claim date:\s*(\d{2}-\d{2}-(?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2})", str(context or ""))
    if date_match:
        context_year = date_match.group(1)[-4:] if date_match.group(1)[2] == "-" else date_match.group(1)[:4]
        if context_year not in normalized_claim:
            entities = " ".join(re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})\b", normalized_claim)[:4])
            historical = f"{entities} {context_year} {normalized_claim[:110]} original records contemporaneous fact check"
            # Reserve one query for claim-period evidence; do not drop the
            # literal claim when four search slots are available.
            search_plan = ([historical] + search_plan[:max_queries - 2] + [normalized_claim])[:max_queries]
    return {
        "normalized_claim": (
            normalized_claim
        ),
        "is_clear": True,
        "reason": (
            "Direct factual claim prepared "
            "without LLM preprocessing in "
            f"{mode.title()} Mode."
        ),
        "search_query": search_plan[0] if search_plan else search_query,
        "verification_queries": search_plan[1:] if search_plan else [normalized_claim],
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

    # For direct claims, a preparation LLM call adds latency but no new evidence.
    # Enable the old Deep Mode behavior only when specifically requested.
    use_llm_preparation = (
        mode == "deep"
        and os.getenv("TRUSTNET_DEEP_LLM_PREPARATION", "0").lower()
        in {"1", "true", "yes", "on"}
    )

    if not use_llm_preparation:
        claim_preparation = _prepare_normal_claim(
            claim,
            context=context,
            mode=mode,
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

    # The first retrieval should usually be enough. Extra rounds were
    # sequential and could consume the time reserved for final LLM analysis.
    # Attempt ONE follow-up only if we found fewer than two source items and
    # there is still a full LLM call worth of time left.
    if mode == "deep" and len(evidence) < 2 and budget.can_start(
        expected_seconds=(
            limits["search_stage_max_seconds"]
            + limits["fetch_stage_max_seconds"]
        ),
        reserve_seconds=(
            limits["llm_call_max_seconds"]
            + limits["finalization_reserve_seconds"]
        ),
    ):
        alternate, followups = _deep_followup_queries(
            normalized_claim=normalized_claim,
            verification_queries=verification_queries,
            suffix=DEEP_REFINEMENT_SUFFIXES[0],
        )
        additional = search_and_retrieve(
            query=normalized_claim,
            alternate_query=alternate,
            verification_queries=followups[:1],
            search_results=search_results,
            top_k=top_k,
            end_date=retrieval_end_date,
            budget=budget,
            mode=mode,
        )
        evidence = _merge_evidence(
            evidence,
            additional,
            max_items=max(top_k + 2, 6),
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

    # A completed search may still omit one essential proposition despite
    # returning four topical URLs.  Give the explicit missing proposition one
    # extra retrieval attempt, then rerun the SAME grounded analysis.  This
    # does not change truth labels from keywords and is bounded by the request
    # deadline, not an unbounded sequence of LLM calls.
    if (not result.get("budget_exhausted")
        and os.getenv("TRUSTNET_RECOVERY", "1").lower() not in {"0", "false", "no", "off"}
        and budget.can_start(expected_seconds=limits["search_stage_max_seconds"]
                             + limits["fetch_stage_max_seconds"],
                             reserve_seconds=limits["finalization_reserve_seconds"]
                             + min(18.0, limits["llm_call_max_seconds"]))):
        followup = recovery_query(normalized_claim, result)
        if followup:
            additional = search_and_retrieve(
                query=followup,
                alternate_query=followup,
                verification_queries=[],
                search_results=search_results,
                top_k=2,
                end_date=retrieval_end_date,
                budget=budget,
                mode=mode,
            )
            augmented = _merge_evidence(
                evidence, additional,
                max_items=max(top_k + 2, 6),
            )
            if len(augmented) > len(evidence) and budget.can_start(
                expected_seconds=0.5,
                reserve_seconds=limits["finalization_reserve_seconds"] + 3.0,
            ):
                recovered_count = len(augmented) - len(evidence)
                retry_result = analyze_claim_with_llm(
                    claim=normalized_claim,
                    evidence=augmented,
                    budget=budget,
                    mode=mode,
                )
                if not retry_result.get("budget_exhausted"):
                    evidence = augmented
                    result = retry_result
                    result["retrieval_recovery"] = {
                        "triggered": True,
                        "query": followup,
                        "new_source_count": recovered_count,
                    }
                else:
                    result["retrieval_recovery"] = {
                        "triggered": True,
                        "query": followup,
                        "outcome": "deadline_preserved_original_analysis",
                    }

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
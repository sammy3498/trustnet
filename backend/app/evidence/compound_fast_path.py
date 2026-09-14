import time

from backend.app.evidence.fast_path import (
    is_simple_atomic_claim,
    try_fast_evidence_analysis,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.authority_recovery import (
    RELIABLE_QUALITIES,
    recover_authority_evidence,
)


SUPPORTED_DIRECTIONS = {
    "supports",
    "contradicts",
}


def append_unique_evidence(
    evidence: list[
        RetrievedEvidence
    ],
    additions: list[
        RetrievedEvidence
    ],
) -> list[
    RetrievedEvidence
]:
    existing_urls = {
        item.url
        for item in evidence
    }

    added = []

    for item in additions:
        if item.url in existing_urls:
            continue

        evidence.append(
            item
        )

        added.append(
            item
        )

        existing_urls.add(
            item.url
        )

    return added


def build_compound_recovery_context(
    evidence: list[
        RetrievedEvidence
    ],
) -> list[
    RetrievedEvidence
]:
    context = []

    for item in evidence:
        if (
            item.evidence_origin
            == "webpage"
            and item.quality
            in RELIABLE_QUALITIES
        ):
            context.append(
                item.model_copy(
                    update={
                        "quality": "low"
                    }
                )
            )
            continue

        context.append(
            item
        )

    return context


def resolve_atomic_subclaim(
    subclaim: str,
    evidence: list[
        RetrievedEvidence
    ],
    end_date: str | None = None,
    deadline: float | None = None,
) -> dict:
    fast_path_ms = 0
    authority_recovery_ms = 0

    probe_start = (
        time.perf_counter()
    )

    analysis = (
        try_fast_evidence_analysis(
            claim=subclaim,
            evidence=evidence,
        )
    )

    fast_path_ms += round(
        (
            time.perf_counter()
            - probe_start
        )
        * 1000
    )

    if analysis is not None:
        return {
            "analysis": analysis,
            "recovered_evidence": [],
            "fast_path_ms": (
                fast_path_ms
            ),
            "authority_recovery_ms": 0,
        }

    recovery_start = (
        time.perf_counter()
    )

    recovery_context = (
        build_compound_recovery_context(
            evidence
        )
    )

    recovered_candidates = (
        recover_authority_evidence(
            claim=subclaim,
            existing_evidence=(
                recovery_context
            ),
            end_date=end_date,
            deadline=deadline,
        )
    )

    authority_recovery_ms += round(
        (
            time.perf_counter()
            - recovery_start
        )
        * 1000
    )

    recovered = (
        append_unique_evidence(
            evidence=evidence,
            additions=(
                recovered_candidates
            ),
        )
    )

    if not recovered:
        return {
            "analysis": None,
            "recovered_evidence": [],
            "fast_path_ms": (
                fast_path_ms
            ),
            "authority_recovery_ms": (
                authority_recovery_ms
            ),
        }

    probe_start = (
        time.perf_counter()
    )

    analysis = (
        try_fast_evidence_analysis(
            claim=subclaim,
            evidence=recovered,
        )
    )

    fast_path_ms += round(
        (
            time.perf_counter()
            - probe_start
        )
        * 1000
    )

    if analysis is None:
        probe_start = (
            time.perf_counter()
        )

        analysis = (
            try_fast_evidence_analysis(
                claim=subclaim,
                evidence=evidence,
            )
        )

        fast_path_ms += round(
            (
                time.perf_counter()
                - probe_start
            )
            * 1000
        )

    return {
        "analysis": analysis,
        "recovered_evidence": (
            recovered
        ),
        "fast_path_ms": (
            fast_path_ms
        ),
        "authority_recovery_ms": (
            authority_recovery_ms
        ),
    }


def get_subclaim_assessment(
    subclaim: str,
    analysis: dict,
) -> dict:
    existing = (
        analysis.get(
            "subclaim_assessments",
            [],
        )
    )

    if existing:
        assessment = dict(
            existing[
                0
            ]
        )

        assessment[
            "subclaim"
        ] = subclaim

        return assessment

    direction = (
        analysis.get(
            "overall_evidence_direction"
        )
    )

    return {
        "subclaim": subclaim,
        "coverage": "complete",
        "direct_support": (
            direction
            == "supports"
        ),
        "direct_contradiction": (
            direction
            == "contradicts"
        ),
        "material_context_conflict": (
            False
        ),
        "status": direction,
        "reasoning": (
            "The atomic subclaim was "
            "resolved by TrustNet's "
            "local evidence fast path."
        ),
    }


def merge_material_caveats(
    analyses: list[dict],
) -> list[str]:
    caveats = []

    for analysis in analyses:
        for caveat in (
            analysis.get(
                "material_caveats",
                [],
            )
        ):
            if caveat not in caveats:
                caveats.append(
                    caveat
                )

    return caveats


def merge_evidence_assessments(
    analyses: list[dict],
) -> list[dict]:
    merged = []

    seen = set()

    for analysis in analyses:
        for assessment in (
            analysis.get(
                "evidence_assessments",
                [],
            )
        ):
            key = (
                assessment.get(
                    "url"
                ),
                assessment.get(
                    "stance"
                ),
                assessment.get(
                    "directness"
                ),
                assessment.get(
                    "qualifier_relation"
                ),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            merged.append(
                dict(
                    assessment
                )
            )

    return merged


def combine_fast_metadata(
    analyses: list[dict],
) -> dict:
    metadata = [
        analysis.get(
            "_fast_path",
            {},
        )
        for analysis in analyses
    ]

    evidence_counts = [
        int(
            item.get(
                "effective_evidence_count",
                0,
            )
        )
        for item in metadata
    ]

    high_counts = [
        int(
            item.get(
                "effective_high_quality_count",
                0,
            )
        )
        for item in metadata
    ]

    medium_counts = [
        int(
            item.get(
                "effective_medium_quality_count",
                0,
            )
        )
        for item in metadata
    ]

    return {
        "route": (
            "local_nli_compound"
        ),
        "effective_evidence_count": (
            min(
                evidence_counts
            )
            if evidence_counts
            else 0
        ),
        "effective_high_quality_count": (
            max(
                high_counts
            )
            if high_counts
            else 0
        ),
        "effective_medium_quality_count": (
            max(
                medium_counts
            )
            if medium_counts
            else 0
        ),
    }


def combine_subclaim_analyses(
    claim: str,
    subclaims: list[str],
    analyses: list[dict],
) -> dict | None:
    if (
        len(
            subclaims
        )
        != len(
            analyses
        )
    ):
        return None

    if not analyses:
        return None

    directions = [
        analysis.get(
            "overall_evidence_direction"
        )
        for analysis in analyses
    ]

    if not all(
        direction
        in SUPPORTED_DIRECTIONS
        for direction
        in directions
    ):
        return None

    if all(
        direction == "supports"
        for direction in directions
    ):
        overall_direction = (
            "supports"
        )

        summary = (
            "All independently verified "
            "atomic components of the "
            "compound claim are supported "
            "by the retrieved evidence."
        )

    else:
        overall_direction = (
            "contradicts"
        )

        summary = (
            "At least one independently "
            "verified atomic component of "
            "the compound claim is "
            "contradicted by the retrieved "
            "evidence."
        )

    subclaim_assessments = [
        get_subclaim_assessment(
            subclaim=subclaim,
            analysis=analysis,
        )
        for subclaim, analysis
        in zip(
            subclaims,
            analyses,
        )
    ]

    return {
        "claim": claim,
        "evidence_assessments": (
            merge_evidence_assessments(
                analyses
            )
        ),
        "subclaim_assessments": (
            subclaim_assessments
        ),
        "overall_evidence_direction": (
            overall_direction
        ),
        "summary": summary,
        "bias_framing": (
            "neutral"
        ),
        "bias_framing_justification": (
            "No material framing concern "
            "was established for the "
            "independently verified simple "
            "factual components."
        ),
        "incentive_conflict": (
            "none_detected"
        ),
        "incentive_conflict_justification": (
            "No incentive or conflict was "
            "established from the supplied "
            "evidence."
        ),
        "material_caveats": (
            merge_material_caveats(
                analyses
            )
        ),
        "_fast_path": (
            combine_fast_metadata(
                analyses
            )
        ),
        "_compound_operator": "and",
    }


def try_compound_fast_evidence_analysis(
    claim: str,
    subclaims: list[str],
    evidence: list[
        RetrievedEvidence
    ],
    end_date: str | None = None,
    deadline: float | None = None,
) -> dict:
    if (
        len(
            subclaims
        )
        < 2
    ):
        return {
            "analysis": None,
            "recovered_evidence": [],
            "fast_path_ms": 0,
            "authority_recovery_ms": 0,
        }

    if not all(
        is_simple_atomic_claim(
            subclaim
        )
        for subclaim in subclaims
    ):
        return {
            "analysis": None,
            "recovered_evidence": [],
            "fast_path_ms": 0,
            "authority_recovery_ms": 0,
        }

    all_recovered = []

    analyses = []

    fast_path_ms = 0

    authority_recovery_ms = 0

    for subclaim in subclaims:
        subclaim_evidence = list(
            evidence
        )

        result = (
            resolve_atomic_subclaim(
                subclaim=subclaim,
                evidence=(
                    subclaim_evidence
                ),
                end_date=end_date,
                deadline=deadline,
            )
        )

        fast_path_ms += int(
            result[
                "fast_path_ms"
            ]
        )

        authority_recovery_ms += int(
            result[
                "authority_recovery_ms"
            ]
        )

        recovered = (
            result[
                "recovered_evidence"
            ]
        )

        append_unique_evidence(
            evidence=all_recovered,
            additions=recovered,
        )

        analysis = (
            result[
                "analysis"
            ]
        )

        if analysis is None:
            return {
                "analysis": None,
                "recovered_evidence": (
                    all_recovered
                ),
                "fast_path_ms": (
                    fast_path_ms
                ),
                "authority_recovery_ms": (
                    authority_recovery_ms
                ),
            }

        analyses.append(
            analysis
        )

    combined = (
        combine_subclaim_analyses(
            claim=claim,
            subclaims=subclaims,
            analyses=analyses,
        )
    )

    return {
        "analysis": combined,
        "recovered_evidence": (
            all_recovered
        ),
        "fast_path_ms": (
            fast_path_ms
        ),
        "authority_recovery_ms": (
            authority_recovery_ms
        ),
    }

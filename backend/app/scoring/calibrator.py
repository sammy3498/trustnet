from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)


BIAS_FRAMING_SCORES = {
    "neutral": 100,
    "minor_concern": 85,
    "material_concern": 50,
    "severe_concern": 15,
}


INCENTIVE_CONFLICT_SCORES = {
    "none_detected": 100,
    "minor_concern": 85,
    "material_concern": 50,
    "severe_concern": 15,
}


def get_evidence_by_url(
    evidence: list[
        RetrievedEvidence
    ],
) -> dict[
    str,
    RetrievedEvidence,
]:
    return {
        item.url: item
        for item in evidence
    }


def collect_direct_publishers(
    evidence_analysis: dict,
    evidence: list[
        RetrievedEvidence
    ],
    target_stance: str,
) -> dict[str, set[str]]:
    evidence_by_url = (
        get_evidence_by_url(
            evidence
        )
    )

    publishers = {
        "high": set(),
        "medium": set(),
        "low": set(),
    }

    for assessment in (
        evidence_analysis.get(
            "evidence_assessments",
            []
        )
    ):
        if (
            assessment.get(
                "stance"
            )
            != target_stance
        ):
            continue

        if (
            assessment.get(
                "directness"
            )
            != "direct"
        ):
            continue

        url = assessment.get(
            "url"
        )

        item = evidence_by_url.get(
            url
        )

        if item is None:
            continue

        publisher = (
            get_publisher_domain(
                item.source
                or item.url
            )
        )

        if not publisher:
            publisher = item.url

        publishers[
            item.quality
        ].add(
            publisher
        )

    return publishers


def calibrate_support_strength(
    publishers: dict[
        str,
        set[str],
    ],
) -> int:
    high_count = len(
        publishers["high"]
    )

    medium_count = len(
        publishers["medium"]
    )

    low_count = len(
        publishers["low"]
    )

    if high_count >= 2:
        return 99

    if (
        high_count >= 1
        and medium_count >= 1
    ):
        return 97

    if high_count >= 1:
        return 95

    if medium_count >= 2:
        return 92

    if medium_count >= 1:
        return 88

    if low_count >= 2:
        return 72

    if low_count >= 1:
        return 68

    return 60


def calibrate_contradiction_strength(
    publishers: dict[
        str,
        set[str],
    ],
) -> int:
    high_count = len(
        publishers["high"]
    )

    medium_count = len(
        publishers["medium"]
    )

    low_count = len(
        publishers["low"]
    )

    if high_count >= 2:
        return 1

    if (
        high_count >= 1
        and medium_count >= 1
    ):
        return 3

    if high_count >= 1:
        return 5

    if medium_count >= 2:
        return 8

    if medium_count >= 1:
        return 12

    if low_count >= 2:
        return 28

    if low_count >= 1:
        return 32

    return 40


def calibrate_evidence_strength(
    evidence_analysis: dict,
    evidence: list[
        RetrievedEvidence
    ],
) -> int:
    direction = (
        evidence_analysis.get(
            "overall_evidence_direction"
        )
    )

    if direction == "supports":
        publishers = (
            collect_direct_publishers(
                evidence_analysis=(
                    evidence_analysis
                ),
                evidence=evidence,
                target_stance="supports",
            )
        )

        return (
            calibrate_support_strength(
                publishers
            )
        )

    if direction == "contradicts":
        publishers = (
            collect_direct_publishers(
                evidence_analysis=(
                    evidence_analysis
                ),
                evidence=evidence,
                target_stance=(
                    "contradicts"
                ),
            )
        )

        return (
            calibrate_contradiction_strength(
                publishers
            )
        )

    return 50


def calibrate_factual_consistency(
    evidence_analysis: dict,
) -> int:
    subclaims = (
        evidence_analysis.get(
            "subclaim_assessments",
            []
        )
    )

    statuses = [
        item.get(
            "status"
        )
        for item in subclaims
        if item.get(
            "status"
        )
    ]

    if not statuses:
        direction = (
            evidence_analysis.get(
                "overall_evidence_direction"
            )
        )

        if direction == "supports":
            return 99

        if direction == "contradicts":
            return 1

        return 50

    if all(
        status == "supports"
        for status in statuses
    ):
        return 99

    if all(
        status == "contradicts"
        for status in statuses
    ):
        return 1

    has_support = (
        "supports" in statuses
    )

    has_contradiction = (
        "contradicts" in statuses
    )

    has_mixed = (
        "mixed" in statuses
    )

    has_insufficient = (
        "insufficient" in statuses
    )

    compound_operator = (
        evidence_analysis.get(
            "_compound_operator"
        )
    )

    if (
        compound_operator == "and"
        and has_contradiction
    ):
        return 5

    if (
        has_mixed
        or (
            has_support
            and has_contradiction
        )
    ):
        return 50

    if (
        has_contradiction
        and has_insufficient
    ):
        return 10

    if (
        has_support
        and has_insufficient
    ):
        return 60

    if has_contradiction:
        return 5

    if has_support:
        return 95

    return 50


def get_relevant_urls(
    evidence_analysis: dict,
) -> set[str]:
    urls = set()

    for assessment in (
        evidence_analysis.get(
            "evidence_assessments",
            []
        )
    ):
        relevance = (
            assessment.get(
                "relevance"
            )
        )

        directness = (
            assessment.get(
                "directness"
            )
        )

        if (
            relevance == "low"
            or directness == "none"
        ):
            continue

        url = assessment.get(
            "url"
        )

        if url:
            urls.add(
                url
            )

    return urls


def calibrate_source_provenance(
    evidence_analysis: dict,
    evidence: list[
        RetrievedEvidence
    ],
) -> int:
    relevant_urls = (
        get_relevant_urls(
            evidence_analysis
        )
    )

    selected_evidence = [
        item
        for item in evidence
        if (
            not relevant_urls
            or item.url
            in relevant_urls
        )
    ]

    publishers = {
        "high": set(),
        "medium": set(),
        "low": set(),
    }

    for item in selected_evidence:
        publisher = (
            get_publisher_domain(
                item.source
                or item.url
            )
        )

        if not publisher:
            publisher = item.url

        publishers[
            item.quality
        ].add(
            publisher
        )

    high_count = len(
        publishers["high"]
    )

    medium_count = len(
        publishers["medium"]
    )

    low_count = len(
        publishers["low"]
    )

    if high_count >= 2:
        return 100

    if (
        high_count >= 1
        and medium_count >= 1
    ):
        return 95

    if high_count >= 1:
        return 90

    if medium_count >= 3:
        return 88

    if medium_count >= 2:
        return 85

    if medium_count >= 1:
        return 75

    if low_count >= 3:
        return 60

    if low_count >= 2:
        return 55

    if low_count >= 1:
        return 50

    return 40


def build_evidence_strength_justification(
    evidence_analysis: dict,
    score: int,
) -> str:
    direction = (
        evidence_analysis.get(
            "overall_evidence_direction",
            "insufficient",
        )
    )

    if direction == "supports":
        return (
            "The supplied evidence "
            "directly supports the claim. "
            "TrustNet calibrated the "
            f"strength to {score} using "
            "the number, independence, "
            "and quality of directly "
            "supporting sources."
        )

    if direction == "contradicts":
        return (
            "The supplied evidence "
            "directly contradicts the "
            "claim. TrustNet calibrated "
            f"the strength to {score} "
            "using the number, "
            "independence, and quality "
            "of directly contradicting "
            "sources."
        )

    if direction == "mixed":
        return (
            "The supplied evidence "
            "contains materially mixed "
            "support and contradiction, "
            "so Evidence Strength is "
            "centered at 50."
        )

    return (
        "The supplied evidence is not "
        "sufficiently complete to "
        "establish a reliable direction, "
        "so Evidence Strength is "
        "centered at 50."
    )


def build_factual_consistency_justification(
    evidence_analysis: dict,
    score: int,
) -> str:
    statuses = [
        item.get(
            "status"
        )
        for item in (
            evidence_analysis.get(
                "subclaim_assessments",
                []
            )
        )
        if item.get(
            "status"
        )
    ]

    if score >= 95:
        return (
            "The material factual "
            "components of the claim are "
            "consistently supported by "
            "the supplied evidence."
        )

    if score <= 10:
        return (
            "One or more material factual "
            "components are directly "
            "contradicted by the supplied "
            "evidence."
        )

    if (
        "mixed" in statuses
        or (
            "supports" in statuses
            and "contradicts"
            in statuses
        )
    ):
        return (
            "Material components are "
            "both supported and "
            "contradicted, producing a "
            "mixed factual-consistency "
            "assessment."
        )

    return (
        "The supplied evidence does not "
        "fully resolve every material "
        "component of the claim."
    )


def build_source_provenance_justification(
    score: int,
) -> str:
    if score >= 95:
        return (
            "The relevant evidence set "
            "contains multiple "
            "independent high-quality or "
            "otherwise strongly reliable "
            "sources."
        )

    if score >= 85:
        return (
            "The relevant evidence set "
            "has strong provenance, but "
            "the highest level of "
            "independent authoritative "
            "corroboration was not "
            "reached."
        )

    if score >= 70:
        return (
            "The evidence includes some "
            "reliable provenance, but "
            "authoritative independent "
            "corroboration is limited."
        )

    return (
        "The available evidence has weak "
        "or limited source provenance."
    )


def build_calibrated_factors(
    evidence_analysis: dict,
    credibility_assessment: dict,
    evidence: list[
        RetrievedEvidence
    ],
) -> dict:
    evidence_strength = (
        calibrate_evidence_strength(
            evidence_analysis=(
                evidence_analysis
            ),
            evidence=evidence,
        )
    )

    factual_consistency = (
        calibrate_factual_consistency(
            evidence_analysis
        )
    )

    source_provenance = (
        calibrate_source_provenance(
            evidence_analysis=(
                evidence_analysis
            ),
            evidence=evidence,
        )
    )

    bias_category = (
        credibility_assessment[
            "bias_framing"
        ]
    )

    incentive_category = (
        credibility_assessment[
            "incentive_conflict"
        ]
    )

    bias_score = (
        BIAS_FRAMING_SCORES[
            bias_category
        ]
    )

    incentive_score = (
        INCENTIVE_CONFLICT_SCORES[
            incentive_category
        ]
    )

    return {
        "evidence_strength": {
            "score": (
                evidence_strength
            ),
            "justification": (
                build_evidence_strength_justification(
                    evidence_analysis,
                    evidence_strength,
                )
            ),
        },
        "factual_consistency": {
            "score": (
                factual_consistency
            ),
            "justification": (
                build_factual_consistency_justification(
                    evidence_analysis,
                    factual_consistency,
                )
            ),
        },
        "source_provenance": {
            "score": (
                source_provenance
            ),
            "justification": (
                build_source_provenance_justification(
                    source_provenance
                )
            ),
        },
        "bias_framing": {
            "score": (
                bias_score
            ),
            "justification": (
                credibility_assessment[
                    "bias_framing_justification"
                ]
            ),
        },
        "incentive_conflict": {
            "score": (
                incentive_score
            ),
            "justification": (
                credibility_assessment[
                    "incentive_conflict_justification"
                ]
            ),
        },
    }

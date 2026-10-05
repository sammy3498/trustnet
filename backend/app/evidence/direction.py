from typing import Literal


EvidenceDirection = Literal[
    "supports",
    "contradicts",
    "mixed",
    "insufficient",
]


def resolve_overall_evidence_direction(
    subclaim_assessments: list[dict],
    fallback_direction: EvidenceDirection,
) -> EvidenceDirection:
    if not subclaim_assessments:
        return fallback_direction

    statuses = {
        item.get("status")
        for item in subclaim_assessments
    }

    statuses.discard(None)

    if "mixed" in statuses:
        return "mixed"

    has_support = "supports" in statuses
    has_contradiction = "contradicts" in statuses
    has_insufficient = "insufficient" in statuses

    if has_support and has_contradiction:
        return "mixed"

    if has_contradiction:
        return "contradicts"

    if has_support and has_insufficient:
        return "insufficient"

    if has_support:
        return "supports"

    return "insufficient"


def align_evidence_strength_score(
    score: int,
    original_direction: EvidenceDirection,
    resolved_direction: EvidenceDirection,
) -> int:
    score = max(0, min(100, int(score)))

    if resolved_direction in {
        "mixed",
        "insufficient",
    }:
        return 50

    if resolved_direction == "supports":
        if score < 50:
            return 100 - score
        return score

    if resolved_direction == "contradicts":
        if score > 50:
            return 100 - score
        return score

    return score

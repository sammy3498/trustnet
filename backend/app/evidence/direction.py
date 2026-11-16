"""Resolve material subclaims using the logical connective in the claim."""
import re
from typing import Literal

EvidenceDirection = Literal["supports", "contradicts", "mixed", "insufficient"]


def resolve_overall_evidence_direction(
    subclaim_assessments: list[dict],
    fallback_direction: EvidenceDirection,
    claim_text: str = "",
) -> EvidenceDirection:
    if not subclaim_assessments:
        return fallback_direction
    statuses = {item.get("status") for item in subclaim_assessments}
    statuses.discard(None)
    if "mixed" in statuses:
        return "mixed"
    # A top-level OR can be satisfied by any true disjunct.  The textual
    # heuristic is deliberately conservative for sentences with mixed logic.
    disjunction = (len(subclaim_assessments) > 1
                   and bool(re.search(r"\bor\b", claim_text, flags=re.I))
                   and not bool(re.search(r"\b(and|but|whereas|while)\b", claim_text, flags=re.I)))
    if disjunction:
        if "supports" in statuses:
            return "supports"
        if "insufficient" in statuses:
            return "insufficient"
        if "contradicts" in statuses:
            return "contradicts"
        return "insufficient"
    # Factual assertions joined by AND/BUT are conjunctive: one established
    # false material assertion refutes the claim as a whole.  Do not call it
    # "mixed" merely because another component is independently true.
    if "contradicts" in statuses:
        return "contradicts"
    if "insufficient" in statuses:
        return "insufficient"
    if "supports" in statuses:
        return "supports"
    return "insufficient"


def align_evidence_strength_score(
    score: int,
    original_direction: EvidenceDirection,
    resolved_direction: EvidenceDirection,
) -> int:
    score = max(0, min(100, int(score)))
    if resolved_direction in {"mixed", "insufficient"}:
        return 50
    if resolved_direction == "supports":
        return max(50, score)
    if resolved_direction == "contradicts":
        return min(49, score)
    return score

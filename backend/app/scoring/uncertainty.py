from backend.app.utils.config import (
    UNCERTAINTY_RULES,
)


def evaluate_uncertainty(
    evidence_count: int,
    high_quality_count: int,
    conflicting_evidence: bool,
    claim_is_clear: bool,
    evidence_direction: str | None = None,
    medium_quality_count: int = 0,
) -> tuple[bool, str, str]:

    if not claim_is_clear:
        return (
            True,
            "high",
            "The claim is too vague or ambiguous to assess reliably.",
        )

    if evidence_direction == "insufficient":
        return (
            True,
            "high",
            "The evidence analysis found insufficient evidence to determine the claim's direction.",
        )

    if (
        evidence_count
        < UNCERTAINTY_RULES[
            "minimum_evidence_count"
        ]
    ):
        return (
            True,
            "high",
            "Insufficient relevant evidence was found.",
        )

    has_high_quality = (
        high_quality_count
        >= UNCERTAINTY_RULES[
            "minimum_high_quality_count"
        ]
    )

    has_medium_consensus = (
        medium_quality_count
        >= UNCERTAINTY_RULES[
            "minimum_medium_quality_count_without_high"
        ]
    )

    if not (
        has_high_quality
        or has_medium_consensus
    ):
        return (
            True,
            "high",
            "No high-quality evidence or sufficient independent medium-quality evidence was found.",
        )

    if (
        conflicting_evidence
        or evidence_direction == "mixed"
    ):
        return (
            False,
            "medium",
            "Reliable evidence is mixed or contains unresolved conflicts.",
        )

    return (
        False,
        "low",
        "Sufficient consistent evidence is available.",
    )
"""Uncertainty is calibrated from *material* evidence, not URL counts."""
from backend.app.utils.config import UNCERTAINTY_RULES


def evaluate_uncertainty(
    evidence_count: int,
    high_quality_count: int,
    conflicting_evidence: bool,
    claim_is_clear: bool,
    evidence_direction: str | None = None,
    medium_quality_count: int = 0,
    mixed_grounded: bool = False,
) -> tuple[bool, str, str]:
    if not claim_is_clear:
        return True, "high", "The claim is ambiguous or lacks an assessable factual proposition."
    if evidence_direction == "mixed":
        if not mixed_grounded or evidence_count <= 0:
            return True, "high", "Mixed evidence direction was not grounded in material cited passages."
        if high_quality_count >= 1 or medium_quality_count >= 1:
            return False, "medium", "Material conflict or misleading context is directly documented by reliable evidence."
        return True, "high", "Material conflict is reported, but independent reliable confirmation is needed."
    if evidence_direction not in {"supports", "contradicts"}:
        return True, "high", "The available material evidence does not establish a settled direction."
    if conflicting_evidence:
        return True, "high", "Directly relevant supporting and contradicting evidence require resolution."
    if evidence_count <= 0:
        return True, "high", "No materially relevant evidence was found."
    # A single decisive primary document can settle a claim, especially an
    # existential counterexample to a universal assertion.
    if high_quality_count >= 1:
        return False, "low", "At least one directly relevant, high-authority source supports the verdict."
    if medium_quality_count >= UNCERTAINTY_RULES["minimum_medium_quality_count_without_high"]:
        return False, "low", "Independent relevant sources corroborate this verdict."
    if medium_quality_count >= 1:
        return False, "medium", "The verdict has one directly relevant medium-authority source."
    return False, "medium", "The verdict relies on lower-authority direct evidence and needs corroboration."

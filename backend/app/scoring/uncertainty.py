from backend.app.utils.config import UNCERTAINTY_RULES


def evaluate_uncertainty(
    evidence_count: int,
    high_quality_count: int,
    conflicting_evidence: bool,
    claim_is_clear: bool,
) -> tuple[bool, str, str]:

    if not claim_is_clear:
        return True, "high", "The claim is too vague or ambiguous to assess reliably."

    if evidence_count < UNCERTAINTY_RULES["minimum_evidence_count"]:
        return True, "high", "Insufficient relevant evidence was found."

    if high_quality_count < UNCERTAINTY_RULES["minimum_high_quality_count"]:
        return True, "high", "No high-quality evidence was found."

    if conflicting_evidence:
        return True, "medium", "Reliable evidence contains unresolved conflicts."

    return False, "low", "Sufficient consistent evidence is available."
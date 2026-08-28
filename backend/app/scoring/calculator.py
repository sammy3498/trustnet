from backend.app.utils.config import FACTOR_WEIGHTS


def calculate_overall_score(
    evidence_strength: int,
    factual_consistency: int,
    source_provenance: int,
    bias_framing: int,
    incentive_conflict: int,
) -> int:
    score = (
        FACTOR_WEIGHTS["evidence_strength"] * evidence_strength
        + FACTOR_WEIGHTS["factual_consistency"] * factual_consistency
        + FACTOR_WEIGHTS["source_provenance"] * source_provenance
        + FACTOR_WEIGHTS["bias_framing"] * bias_framing
        + FACTOR_WEIGHTS["incentive_conflict"] * incentive_conflict
    )

    return round(score)


def get_label(score: int, uncertainty_gate: bool = False) -> str:
    if uncertainty_gate:
        return "Insufficient Evidence"

    if score >= 80:
        return "Highly Credible"
    elif score >= 60:
        return "Moderately Credible"
    elif score >= 40:
        return "Low Credibility"
    else:
        return "Not Credible"
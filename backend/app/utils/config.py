FACTOR_WEIGHTS = {
    "evidence_strength": 0.40,
    "factual_consistency": 0.35,
    "source_provenance": 0.10,
    "bias_framing": 0.08,
    "incentive_conflict": 0.07,
}

SCORE_THRESHOLDS = {
    "highly_credible": 80,
    "moderately_credible": 60,
    "low_credibility": 40,
}

UNCERTAINTY_RULES = {
    "minimum_evidence_count": 2,
    "minimum_high_quality_count": 1,
}

SUPPORTED_INPUT_TYPES = {
    "claim",
    "text",
    "url",
}

SUPPORTED_MODES = {
    "normal",
    "deep",
}
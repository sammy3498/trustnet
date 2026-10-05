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
    "minimum_medium_quality_count_without_high": 2,
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


ANALYSIS_TIME_LIMITS = {
    "normal": {
        "target_seconds": 20.0,
        "soft_limit_seconds": 52.0,
        "hard_limit_seconds": 60.0,
        "finalization_reserve_seconds": 4.0,
        "claim_preparation_max_seconds": 6.0,
        "search_stage_max_seconds": 6.0,
        "fetch_stage_max_seconds": 5.0,
        "llm_call_max_seconds": 18.0,
        "max_queries": 3,
        "search_results": 3,
        "top_k": 3,
    },
    "deep": {
        "target_seconds": 55.0,
        "soft_limit_seconds": 56.0,
        "hard_limit_seconds": 60.0,
        "finalization_reserve_seconds": 4.0,
        "claim_preparation_max_seconds": 8.0,
        "search_stage_max_seconds": 10.0,
        "fetch_stage_max_seconds": 12.0,
        "llm_call_max_seconds": 22.0,
        "max_queries": 4,
        "search_results": 8,
        "top_k": 5,
    },
}


def get_mode_limits(
    mode: str,
) -> dict:
    if mode not in ANALYSIS_TIME_LIMITS:
        raise ValueError(
            f"Unsupported analysis mode: {mode}"
        )

    return (
        ANALYSIS_TIME_LIMITS[
            mode
        ].copy()
    )
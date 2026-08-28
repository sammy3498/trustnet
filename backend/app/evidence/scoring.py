from typing import Literal


Stance = Literal["supports", "contradicts", "mixed", "neutral"]
Quality = Literal["high", "medium", "low"]


def stance_score(stance: Stance) -> int:
    scores = {
        "supports": 100,
        "mixed": 50,
        "neutral": 50,
        "contradicts": 0,
    }

    return scores[stance]


def quality_weight(quality: Quality) -> float:
    weights = {
        "high": 1.0,
        "medium": 0.6,
        "low": 0.3,
    }

    return weights[quality]


def weighted_evidence_score(
    stance: Stance,
    quality: Quality,
) -> float:
    return stance_score(stance) * quality_weight(quality)
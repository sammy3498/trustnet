from backend.app.retrieval.hybrid import (
    normalize_scores,
    quality_score,
)


def test_normalize_scores():
    result = normalize_scores([2.0, 4.0, 6.0])

    assert result == [0.0, 0.5, 1.0]


def test_normalize_equal_scores():
    result = normalize_scores([3.0, 3.0, 3.0])

    assert result == [1.0, 1.0, 1.0]


def test_normalize_empty_scores():
    assert normalize_scores([]) == []


def test_quality_score():
    assert quality_score("high") == 1.0
    assert quality_score("medium") == 0.5
    assert quality_score("low") == 0.0
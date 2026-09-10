import pytest

from eval.metrics.classification import (
    LABEL_ORDER,
    build_confusion_matrix,
    calculate_classification_metrics,
)


def test_perfect_predictions():
    gold = list(
        LABEL_ORDER
    )

    result = (
        calculate_classification_metrics(
            gold,
            gold,
        )
    )

    assert result[
        "accuracy"
    ] == 1.0

    assert result[
        "macro_f1"
    ] == 1.0


def test_confusion_matrix_counts():
    gold = [
        "Supported",
        "Supported",
        "Refuted",
    ]

    predicted = [
        "Supported",
        "Refuted",
        "Refuted",
    ]

    matrix = (
        build_confusion_matrix(
            gold,
            predicted,
        )
    )

    assert (
        matrix[
            "Supported"
        ][
            "Supported"
        ]
        == 1
    )

    assert (
        matrix[
            "Supported"
        ][
            "Refuted"
        ]
        == 1
    )

    assert (
        matrix[
            "Refuted"
        ][
            "Refuted"
        ]
        == 1
    )


def test_zero_prediction_class():
    gold = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]

    predicted = [
        "Supported",
        "Supported",
        "Supported",
        "Supported",
    ]

    result = (
        calculate_classification_metrics(
            gold,
            predicted,
        )
    )

    assert (
        result[
            "per_class"
        ][
            "Conflicting Evidence/Cherrypicking"
        ][
            "precision"
        ]
        == 0.0
    )


def test_rejects_different_lengths():
    with pytest.raises(
        ValueError
    ):
        build_confusion_matrix(
            ["Supported"],
            [],
        )


def test_rejects_unknown_label():
    with pytest.raises(
        ValueError
    ):
        build_confusion_matrix(
            ["Supported"],
            ["Unknown"],
        )


def test_rejects_empty_metrics():
    with pytest.raises(
        ValueError
    ):
        calculate_classification_metrics(
            [],
            [],
        )
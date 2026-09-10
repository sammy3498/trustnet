LABEL_ORDER = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)


def build_confusion_matrix(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    if len(gold_labels) != len(
        predicted_labels
    ):
        raise ValueError(
            "Gold and predicted label "
            "counts must match."
        )

    matrix = {
        gold: {
            predicted: 0
            for predicted in LABEL_ORDER
        }
        for gold in LABEL_ORDER
    }

    for gold, predicted in zip(
        gold_labels,
        predicted_labels,
    ):
        if gold not in LABEL_ORDER:
            raise ValueError(
                f"Unknown gold label: {gold}"
            )

        if predicted not in LABEL_ORDER:
            raise ValueError(
                "Unknown predicted label: "
                f"{predicted}"
            )

        matrix[
            gold
        ][
            predicted
        ] += 1

    return matrix


def calculate_class_metrics(
    matrix: dict,
) -> dict:
    metrics = {}

    for label in LABEL_ORDER:
        true_positive = (
            matrix[
                label
            ][
                label
            ]
        )

        false_positive = sum(
            matrix[
                other
            ][
                label
            ]
            for other in LABEL_ORDER
            if other != label
        )

        false_negative = sum(
            matrix[
                label
            ][
                other
            ]
            for other in LABEL_ORDER
            if other != label
        )

        support = sum(
            matrix[
                label
            ].values()
        )

        precision_denominator = (
            true_positive
            + false_positive
        )

        recall_denominator = (
            true_positive
            + false_negative
        )

        precision = (
            true_positive
            / precision_denominator
            if precision_denominator
            else 0.0
        )

        recall = (
            true_positive
            / recall_denominator
            if recall_denominator
            else 0.0
        )

        f1_denominator = (
            precision
            + recall
        )

        f1 = (
            2
            * precision
            * recall
            / f1_denominator
            if f1_denominator
            else 0.0
        )

        metrics[
            label
        ] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return metrics


def calculate_classification_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    if not gold_labels:
        raise ValueError(
            "At least one prediction "
            "is required."
        )

    matrix = (
        build_confusion_matrix(
            gold_labels,
            predicted_labels,
        )
    )

    per_class = (
        calculate_class_metrics(
            matrix
        )
    )

    correct = sum(
        gold == predicted
        for gold, predicted in zip(
            gold_labels,
            predicted_labels,
        )
    )

    accuracy = (
        correct
        / len(
            gold_labels
        )
    )

    macro_precision = sum(
        per_class[
            label
        ][
            "precision"
        ]
        for label in LABEL_ORDER
    ) / len(
        LABEL_ORDER
    )

    macro_recall = sum(
        per_class[
            label
        ][
            "recall"
        ]
        for label in LABEL_ORDER
    ) / len(
        LABEL_ORDER
    )

    macro_f1 = sum(
        per_class[
            label
        ][
            "f1"
        ]
        for label in LABEL_ORDER
    ) / len(
        LABEL_ORDER
    )

    return {
        "count": len(
            gold_labels
        ),
        "correct": correct,
        "accuracy": accuracy,
        "macro_precision": (
            macro_precision
        ),
        "macro_recall": (
            macro_recall
        ),
        "macro_f1": (
            macro_f1
        ),
        "per_class": (
            per_class
        ),
        "confusion_matrix": (
            matrix
        ),
    }
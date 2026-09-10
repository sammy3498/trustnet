import argparse
import json
from pathlib import Path
from statistics import mean

from eval.metrics.classification import (
    LABEL_ORDER,
    calculate_classification_metrics,
)


DEFAULT_INPUT = Path(
    "eval/results/"
    "averitec_stratified_"
    "averitec_standard_"
    "n20_seed42.json"
)


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    if not values:
        return None

    ordered = sorted(
        values
    )

    if len(ordered) == 1:
        return float(
            ordered[0]
        )

    position = (
        (len(ordered) - 1)
        * percentile_value
    )

    lower = int(
        position
    )

    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = (
        position - lower
    )

    return (
        ordered[lower]
        + (
            ordered[upper]
            - ordered[lower]
        )
        * fraction
    )


def build_latency_metrics(
    records: list[dict],
) -> dict:
    all_latencies = [
        float(
            record[
                "analysis_latency_ms"
            ]
        )
        for record in records
        if record.get(
            "analysis_latency_ms"
        )
        is not None
    ]

    llm_latencies = [
        value
        for value in all_latencies
        if value > 0
    ]

    def summarize(
        values: list[float],
    ) -> dict:
        if not values:
            return {
                "count": 0,
                "mean_ms": None,
                "p50_ms": None,
                "p95_ms": None,
                "min_ms": None,
                "max_ms": None,
            }

        return {
            "count": len(values),
            "mean_ms": mean(
                values
            ),
            "p50_ms": percentile(
                values,
                0.50,
            ),
            "p95_ms": percentile(
                values,
                0.95,
            ),
            "min_ms": min(
                values
            ),
            "max_ms": max(
                values
            ),
        }

    return {
        "all_completed": summarize(
            all_latencies
        ),
        "llm_calls_only": summarize(
            llm_latencies
        ),
    }


def build_abstention_metrics(
    records: list[dict],
) -> dict:
    nei = (
        "Not Enough Evidence"
    )

    operational_nei = sum(
        record.get(
            "operational_prediction"
        )
        == nei
        for record in records
    )

    veracity_nei = sum(
        record.get(
            "predicted_label"
        )
        == nei
        for record in records
    )

    additional_abstentions = sum(
        record.get(
            "predicted_label"
        )
        != nei
        and record.get(
            "operational_prediction"
        )
        == nei
        for record in records
    )

    gate_triggered = sum(
        bool(
            record.get(
                "uncertainty",
                {},
            ).get(
                "gate_triggered"
            )
        )
        for record in records
    )

    correct_to_incorrect = sum(
        bool(
            record.get(
                "correct"
            )
        )
        and not bool(
            record.get(
                "operational_correct"
            )
        )
        for record in records
    )

    incorrect_to_correct = sum(
        not bool(
            record.get(
                "correct"
            )
        )
        and bool(
            record.get(
                "operational_correct"
            )
        )
        for record in records
    )

    count = len(
        records
    )

    return {
        "count": count,
        "veracity_nei_count": (
            veracity_nei
        ),
        "operational_nei_count": (
            operational_nei
        ),
        "operational_nei_rate": (
            operational_nei / count
            if count
            else 0.0
        ),
        "additional_abstentions": (
            additional_abstentions
        ),
        "uncertainty_gate_triggered": (
            gate_triggered
        ),
        "correct_to_incorrect": (
            correct_to_incorrect
        ),
        "incorrect_to_correct": (
            incorrect_to_correct
        ),
    }


def predicted_distribution(
    records: list[dict],
    field: str,
) -> dict:
    return {
        label: sum(
            record.get(
                field
            )
            == label
            for record in records
        )
        for label in LABEL_ORDER
    }


def print_classification_report(
    title: str,
    metrics: dict,
):
    print()

    print(
        f"===== {title} ====="
    )

    print(
        "Count:",
        metrics[
            "count"
        ],
    )

    print(
        "Correct:",
        metrics[
            "correct"
        ],
    )

    print(
        "Accuracy:",
        f"{metrics['accuracy']:.3f}",
    )

    print(
        "Macro precision:",
        f"{metrics['macro_precision']:.3f}",
    )

    print(
        "Macro recall:",
        f"{metrics['macro_recall']:.3f}",
    )

    print(
        "Macro F1:",
        f"{metrics['macro_f1']:.3f}",
    )

    print()

    print(
        "PER-CLASS METRICS"
    )

    for label in LABEL_ORDER:
        item = (
            metrics[
                "per_class"
            ][
                label
            ]
        )

        print(
            label,
        )

        print(
            "  Precision:",
            f"{item['precision']:.3f}",
        )

        print(
            "  Recall:",
            f"{item['recall']:.3f}",
        )

        print(
            "  F1:",
            f"{item['f1']:.3f}",
        )

        print(
            "  Support:",
            item[
                "support"
            ],
        )

    print()

    print(
        "CONFUSION MATRIX"
    )

    print(
        "Rows = gold, "
        "columns = predicted"
    )

    header = [
        "GOLD",
        *LABEL_ORDER,
    ]

    print(
        " | ".join(
            header
        )
    )

    for gold in LABEL_ORDER:
        row = [
            gold,
            *[
                str(
                    metrics[
                        "confusion_matrix"
                    ][
                        gold
                    ][
                        predicted
                    ]
                )
                for predicted
                in LABEL_ORDER
            ],
        ]

        print(
            " | ".join(
                row
            )
        )


def run(
    input_path: Path,
):
    if not input_path.exists():
        raise FileNotFoundError(
            f"Result file not found: "
            f"{input_path}"
        )

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(
            file
        )

    if not isinstance(
        results,
        list,
    ):
        raise ValueError(
            "Result file must contain "
            "a JSON list."
        )

    successful = [
        record
        for record in results
        if (
            "predicted_label"
            in record
            and (
                "operational_prediction"
                in record
            )
        )
    ]

    errors = [
        record
        for record in results
        if "error" in record
    ]

    if not successful:
        raise ValueError(
            "No completed predictions "
            "were found."
        )

    gold = [
        record[
            "gold_label"
        ]
        for record in successful
    ]

    veracity_predictions = [
        record[
            "predicted_label"
        ]
        for record in successful
    ]

    operational_predictions = [
        record[
            "operational_prediction"
        ]
        for record in successful
    ]

    veracity_metrics = (
        calculate_classification_metrics(
            gold,
            veracity_predictions,
        )
    )

    operational_metrics = (
        calculate_classification_metrics(
            gold,
            operational_predictions,
        )
    )

    abstention_metrics = (
        build_abstention_metrics(
            successful
        )
    )

    latency_metrics = (
        build_latency_metrics(
            successful
        )
    )

    print(
        "===== AVERITEC STRATIFIED "
        "EVALUATION REPORT ====="
    )

    print(
        f"Input: {input_path}"
    )

    print(
        "Records:",
        len(
            results
        ),
    )

    print(
        "Completed:",
        len(
            successful
        ),
    )

    print(
        "Errors:",
        len(
            errors
        ),
    )

    print_classification_report(
        "VERACITY",
        veracity_metrics,
    )

    print()

    print(
        "VERACITY PREDICTION "
        "DISTRIBUTION:"
    )

    print(
        predicted_distribution(
            successful,
            "predicted_label",
        )
    )

    print_classification_report(
        "OPERATIONAL",
        operational_metrics,
    )

    print()

    print(
        "OPERATIONAL PREDICTION "
        "DISTRIBUTION:"
    )

    print(
        predicted_distribution(
            successful,
            "operational_prediction",
        )
    )

    print()

    print(
        "===== ABSTENTION ====="
    )

    for key, value in (
        abstention_metrics.items()
    ):
        if key == (
            "operational_nei_rate"
        ):
            print(
                f"{key}: "
                f"{value:.3f}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print()

    print(
        "===== LATENCY ====="
    )

    for group, values in (
        latency_metrics.items()
    ):
        print(
            group
        )

        for key, value in (
            values.items()
        ):
            if (
                isinstance(
                    value,
                    float,
                )
                and key != "count"
            ):
                print(
                    f"  {key}: "
                    f"{value:.1f}"
                )
            else:
                print(
                    f"  {key}: "
                    f"{value}"
                )

    report = {
        "input": str(
            input_path
        ),
        "records": len(
            results
        ),
        "completed": len(
            successful
        ),
        "errors": len(
            errors
        ),
        "veracity": (
            veracity_metrics
        ),
        "operational": (
            operational_metrics
        ),
        "veracity_distribution": (
            predicted_distribution(
                successful,
                "predicted_label",
            )
        ),
        "operational_distribution": (
            predicted_distribution(
                successful,
                "operational_prediction",
            )
        ),
        "abstention": (
            abstention_metrics
        ),
        "latency": (
            latency_metrics
        ),
    }

    output_path = (
        input_path.with_name(
            input_path.stem
            + "_report.json"
        )
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()

    print(
        f"REPORT SAVED: {output_path}"
    )


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    args = parser.parse_args()

    run(
        input_path=args.input,
    )
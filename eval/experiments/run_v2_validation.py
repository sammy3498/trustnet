import argparse
import json
import time
from collections import Counter
from pathlib import Path

from backend.app.llm.evidence_analysis import (
    analyze_evidence_with_llm,
)
from eval.loaders.averitec import (
    load_averitec,
)
from eval.sampling.averitec import (
    AVERITEC_LABEL_ORDER,
)
from eval.snapshots.averitec import (
    evidence_from_snapshot,
    get_snapshot_path,
)


VALIDATION_MANIFEST_PATH = Path(
    "eval/snapshots/averitec_stratified/"
    "train_v2_validation_pool.json"
)

DEVELOPMENT_MANIFEST_PATH = Path(
    "eval/snapshots/averitec_stratified/"
    "train_v2_development_pool.json"
)

DATASET_PATH = Path(
    "eval/datasets/averitec/train.json"
)

SNAPSHOT_BASE = Path(
    "eval/snapshots/averitec_stratified/"
    "train/v2_validation"
)

RESULT_ROOT = Path(
    "eval/results"
)

RAW_RESULT_ROOT = (
    RESULT_ROOT
    / "trustnet_v2_validation_raw"
)

EXPECTED_SAMPLE_SIZE = 80
EXPECTED_PER_CLASS = 20
EXPECTED_SEED = 4202

LABELS = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

DIRECTION_TO_LABEL = {
    "supports": "Supported",
    "contradicts": "Refuted",
    "insufficient": "Not Enough Evidence",
    "mixed": (
        "Conflicting Evidence/"
        "Cherrypicking"
    ),
}


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    data,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary.replace(path)


def load_index_manifest(
    path: Path,
    expected_size: int | None = None,
):
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    manifest = load_json(path)

    examples = manifest.get(
        "examples"
    )

    if not isinstance(
        examples,
        list,
    ):
        raise ValueError(
            "Manifest does not contain "
            "an examples list."
        )

    if (
        expected_size is not None
        and len(examples)
        != expected_size
    ):
        raise ValueError(
            "Unexpected manifest size."
        )

    indices = []

    for record in examples:
        if not isinstance(
            record,
            dict,
        ):
            raise ValueError(
                "Manifest example must "
                "be an object."
            )

        index = record.get(
            "index"
        )

        if not isinstance(
            index,
            int,
        ):
            raise ValueError(
                "Every manifest example "
                "must contain an integer "
                "index."
            )

        indices.append(index)

    if len(set(indices)) != len(
        indices
    ):
        raise ValueError(
            "Manifest contains duplicate "
            "indices."
        )

    return manifest, examples


def load_validation_records():
    manifest, records = (
        load_index_manifest(
            VALIDATION_MANIFEST_PATH,
            expected_size=(
                EXPECTED_SAMPLE_SIZE
            ),
        )
    )

    if manifest.get(
        "split"
    ) != "train":
        raise ValueError(
            "Validation split must "
            "be train."
        )

    if manifest.get(
        "seed"
    ) != EXPECTED_SEED:
        raise ValueError(
            "Validation seed mismatch."
        )

    if manifest.get(
        "per_class"
    ) != EXPECTED_PER_CLASS:
        raise ValueError(
            "Validation per_class "
            "mismatch."
        )

    if manifest.get(
        "sample_size"
    ) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            "Validation sample_size "
            "mismatch."
        )

    return manifest, records


def verify_development_disjoint(
    validation_records: list[dict],
):
    _, development_records = (
        load_index_manifest(
            DEVELOPMENT_MANIFEST_PATH
        )
    )

    validation_indices = {
        item["index"]
        for item
        in validation_records
    }

    development_indices = {
        item["index"]
        for item
        in development_records
    }

    if (
        validation_indices
        & development_indices
    ):
        raise ValueError(
            "Validation/development "
            "overlap detected."
        )


def load_dataset_metadata():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "AVeriTeC train dataset "
            f"not found: {DATASET_PATH}"
        )

    dataset = load_averitec(
        DATASET_PATH,
        skip_invalid_claims=True,
    )

    dataset_by_index = {
        example.index: example
        for example in dataset
    }

    raw_dataset = load_json(
        DATASET_PATH
    )

    return (
        dataset_by_index,
        raw_dataset,
    )


def validate_against_dataset(
    records: list[dict],
    dataset_by_index: dict,
):
    labels = []

    for record in records:
        example = (
            dataset_by_index.get(
                record["index"]
            )
        )

        if example is None:
            raise ValueError(
                "Validation index not "
                "found in source dataset."
            )

        labels.append(
            example.label
        )

    counts = Counter(labels)

    if set(counts) != set(
        AVERITEC_LABEL_ORDER
    ):
        raise ValueError(
            "Derived label set does "
            "not match AVeriTeC."
        )

    if any(
        counts[label]
        != EXPECTED_PER_CLASS
        for label
        in AVERITEC_LABEL_ORDER
    ):
        raise ValueError(
            "Derived validation labels "
            "are not balanced."
        )


def get_gold_metadata(
    index: int,
    dataset_by_index: dict,
    raw_dataset: list,
):
    example = dataset_by_index.get(
        index
    )

    if example is None:
        raise ValueError(
            "Validation index missing "
            "from source dataset."
        )

    if (
        index < 0
        or index >= len(
            raw_dataset
        )
    ):
        raise ValueError(
            "Raw dataset index out "
            "of range."
        )

    raw_record = raw_dataset[
        index
    ]

    if not isinstance(
        raw_record,
        dict,
    ):
        raise ValueError(
            "Raw AVeriTeC record "
            "is invalid."
        )

    required_reannotation = bool(
        raw_record.get(
            "required_reannotation",
            False,
        )
    )

    return (
        example.label,
        required_reannotation,
    )


def get_result_path(
    protocol: str,
) -> Path:
    return (
        RESULT_ROOT
        / (
            "trustnet_v2_validation_"
            f"{protocol}.json"
        )
    )


def get_summary_path(
    protocol: str,
) -> Path:
    return (
        RESULT_ROOT
        / (
            "trustnet_v2_validation_"
            f"{protocol}_summary.json"
        )
    )


def get_raw_result_path(
    protocol: str,
    index: int,
) -> Path:
    return (
        RAW_RESULT_ROOT
        / protocol
        / f"{index}.json"
    )


def load_existing(
    protocol: str,
) -> list[dict]:
    path = get_result_path(
        protocol
    )

    if not path.exists():
        return []

    data = load_json(path)

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Existing validation "
            "result file is not a list."
        )

    return data


def load_snapshot(
    protocol: str,
    index: int,
):
    path = get_snapshot_path(
        SNAPSHOT_BASE,
        protocol,
        index,
    )

    if not path.exists():
        raise FileNotFoundError(
            "A required validation "
            "snapshot is missing."
        )

    snapshot = load_json(path)

    if snapshot.get(
        "example_index"
    ) != index:
        raise ValueError(
            "Snapshot index mismatch."
        )

    return snapshot


def snapshot_exists(
    protocol: str,
    index: int,
) -> bool:
    return get_snapshot_path(
        SNAPSHOT_BASE,
        protocol,
        index,
    ).exists()


def build_record(
    *,
    manifest_record: dict,
    snapshot: dict,
    gold_label: str,
    required_reannotation: bool,
    analysis: dict | None,
    latency_ms: int,
):
    index = manifest_record[
        "index"
    ]

    snapshot_gold = snapshot.get(
        "gold_label"
    )

    if (
        snapshot_gold is not None
        and snapshot_gold
        != gold_label
    ):
        raise ValueError(
            "Snapshot/source gold "
            "label mismatch."
        )

    claim = snapshot[
        "claim"
    ]

    evidence = evidence_from_snapshot(
        snapshot
    )

    if analysis is None:
        direction = "insufficient"

        prediction = (
            "Not Enough Evidence"
        )

        llm_direction = None
        direction_was_resolved = False
        subclaims = []
        assessments = []
    else:
        direction = analysis[
            "overall_evidence_direction"
        ]

        if direction not in (
            DIRECTION_TO_LABEL
        ):
            raise ValueError(
                "Unexpected evidence "
                f"direction: {direction}"
            )

        prediction = (
            DIRECTION_TO_LABEL[
                direction
            ]
        )

        llm_direction = (
            analysis.get(
                "llm_overall_"
                "evidence_direction"
            )
        )

        direction_was_resolved = (
            analysis.get(
                "direction_was_resolved",
                False,
            )
        )

        subclaims = analysis.get(
            "subclaim_assessments",
            [],
        )

        assessments = analysis.get(
            "evidence_assessments",
            [],
        )

    return {
        "dataset": "AVeriTeC",
        "dataset_split": "train",
        "evaluation_protocol": (
            "trustnet_v2_validation_"
            "evidence_only"
        ),
        "retrieval_protocol": (
            snapshot.get(
                "evaluation_protocol"
            )
        ),
        "sample_id": (
            "v2_validation_n80_"
            "seed4202"
        ),
        "index": index,
        "claim": claim,
        "gold_label": gold_label,
        "required_reannotation": (
            required_reannotation
        ),
        "predicted_label": (
            prediction
        ),
        "operational_prediction": (
            prediction
        ),
        "correct": (
            prediction
            == gold_label
        ),
        "operational_correct": (
            prediction
            == gold_label
        ),
        "evidence_direction": (
            direction
        ),
        "llm_evidence_direction": (
            llm_direction
        ),
        "direction_was_resolved": (
            direction_was_resolved
        ),
        "subclaim_assessments": (
            subclaims
        ),
        "evidence_assessments": (
            assessments
        ),
        "uncertainty": {
            "gate_triggered": False,
            "level": "none",
            "reason": (
                "Evidence-only frozen "
                "v2 validation."
            ),
        },
        "analysis_latency_ms": (
            latency_ms
        ),
        "llm_timings": {
            "evidence_analysis_ms": (
                latency_ms
            ),
        },
        "sources": [
            {
                "title": item.title,
                "source": item.source,
                "url": item.url,
                "quality": (
                    item.quality
                ),
                "origin": (
                    item.evidence_origin
                ),
                "published_date": (
                    item.published_date
                ),
            }
            for item in evidence
        ],
    }


def count_validation_state(
    records: list[dict],
    protocol: str,
):
    existing = load_existing(
        protocol
    )

    completed_indices = {
        item["index"]
        for item in existing
        if (
            isinstance(
                item,
                dict,
            )
            and "predicted_label"
            in item
        )
    }

    snapshots_ready = 0
    snapshots_missing = 0
    zero_evidence = 0
    raw_recoverable = 0
    new_llm_needed = 0

    for record in records:
        index = record[
            "index"
        ]

        if not snapshot_exists(
            protocol,
            index,
        ):
            snapshots_missing += 1
            continue

        snapshots_ready += 1

        if index in completed_indices:
            continue

        snapshot = load_snapshot(
            protocol,
            index,
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        if not evidence:
            zero_evidence += 1
            continue

        raw_path = (
            get_raw_result_path(
                protocol,
                index,
            )
        )

        if raw_path.exists():
            raw_recoverable += 1
        else:
            new_llm_needed += 1

    return {
        "existing": existing,
        "completed_indices": (
            completed_indices
        ),
        "snapshots_ready": (
            snapshots_ready
        ),
        "snapshots_missing": (
            snapshots_missing
        ),
        "zero_evidence": (
            zero_evidence
        ),
        "raw_recoverable": (
            raw_recoverable
        ),
        "new_llm_needed": (
            new_llm_needed
        ),
    }


def compute_metrics(
    records: list[dict],
):
    total = len(records)

    if total == 0:
        return {
            "n": 0,
            "accuracy": None,
            "macro_f1": None,
        }

    correct = sum(
        item["predicted_label"]
        == item["gold_label"]
        for item in records
    )

    per_class = {}
    f1_values = []

    for label in LABELS:
        tp = sum(
            item["gold_label"] == label
            and item[
                "predicted_label"
            ] == label
            for item in records
        )

        fp = sum(
            item["gold_label"] != label
            and item[
                "predicted_label"
            ] == label
            for item in records
        )

        fn = sum(
            item["gold_label"] == label
            and item[
                "predicted_label"
            ] != label
            for item in records
        )

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
            if precision
            + recall
            else 0.0
        )

        f1_values.append(f1)

        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(
                item["gold_label"]
                == label
                for item in records
            ),
        }

    return {
        "n": total,
        "correct": correct,
        "accuracy": (
            correct / total
        ),
        "macro_f1": (
            sum(f1_values)
            / len(LABELS)
        ),
        "per_class": per_class,
    }


def dry_run(
    records: list[dict],
    protocol: str,
    max_attempts: int,
):
    state = count_validation_state(
        records,
        protocol,
    )

    completed = len(
        state[
            "completed_indices"
        ]
    )

    print(
        "===== V2 VALIDATION "
        "DRY RUN ====="
    )
    print(
        "PROTOCOL:",
        protocol,
    )
    print(
        "VALIDATION CASES:",
        len(records),
    )
    print(
        "VALIDATION SEED:",
        EXPECTED_SEED,
    )
    print(
        "LABEL BALANCE VALID:",
        True,
    )
    print(
        "DEV/VALIDATION "
        "DISJOINT:",
        True,
    )
    print(
        "PRIMARY METRICS:",
        "full accuracy + macro-F1",
    )
    print(
        "PREDECLARED SECONDARY "
        "METRICS:",
        "required_reannotation="
        "False accuracy + macro-F1",
    )
    print(
        "SNAPSHOTS READY:",
        state[
            "snapshots_ready"
        ],
    )
    print(
        "SNAPSHOTS MISSING:",
        state[
            "snapshots_missing"
        ],
    )
    print(
        "COMPLETED RESULTS:",
        completed,
    )
    print(
        "PENDING ZERO-EVIDENCE "
        "CASES:",
        state[
            "zero_evidence"
        ],
    )
    print(
        "RAW RESULTS "
        "RECOVERABLE:",
        state[
            "raw_recoverable"
        ],
    )
    print(
        "NEW LLM CALLS STILL "
        "NEEDED:",
        state[
            "new_llm_needed"
        ],
    )
    print(
        "MAX NEW LLM CALLS "
        "REQUESTED:",
        max_attempts,
    )
    print(
        "MAX NEW LLM CALLS "
        "THIS RUN:",
        min(
            max_attempts,
            state[
                "new_llm_needed"
            ],
        ),
    )
    print(
        "TAVILY CALLS:",
        0,
    )
    print(
        "CLAIMS OR INDICES "
        "PRINTED:",
        0,
    )


def save_progress(
    records: list[dict],
    records_by_index: dict,
    result_path: Path,
):
    ordered = [
        records_by_index[
            item["index"]
        ]
        for item in records
        if item["index"]
        in records_by_index
    ]

    save_json(
        result_path,
        ordered,
    )

    return ordered


def run(
    records: list[dict],
    dataset_by_index: dict,
    raw_dataset: list,
    protocol: str,
    max_attempts: int,
    confirm_live: bool,
):
    if max_attempts <= 0:
        raise ValueError(
            "--max-attempts must be "
            "greater than zero for "
            "a live run."
        )

    if not confirm_live:
        raise ValueError(
            "Live validation requires "
            "--confirm-live."
        )

    state = count_validation_state(
        records,
        protocol,
    )

    existing = state[
        "existing"
    ]

    records_by_index = {
        item["index"]: item
        for item in existing
        if (
            isinstance(
                item,
                dict,
            )
            and isinstance(
                item.get("index"),
                int,
            )
        )
    }

    attempts = 0
    recovered = 0
    zero_evidence_completed = 0
    missing_skipped = 0
    llm_cap_skipped = 0
    stopped_on_error = False

    result_path = get_result_path(
        protocol
    )

    print(
        "===== V2 FROZEN "
        "VALIDATION ====="
    )
    print(
        "PROTOCOL:",
        protocol,
    )
    print(
        "VALIDATION CASES:",
        len(records),
    )
    print(
        "SNAPSHOTS READY AT START:",
        state[
            "snapshots_ready"
        ],
    )
    print(
        "SNAPSHOTS MISSING AT START:",
        state[
            "snapshots_missing"
        ],
    )
    print(
        "MAX NEW LLM CALLS:",
        max_attempts,
    )
    print(
        "CLAIMS OR INDICES "
        "PRINTED:",
        0,
    )

    for position, manifest_record in (
        enumerate(
            records,
            start=1,
        )
    ):
        index = manifest_record[
            "index"
        ]

        existing_record = (
            records_by_index.get(
                index
            )
        )

        if (
            existing_record
            and "predicted_label"
            in existing_record
        ):
            print(
                f"[{position}/"
                f"{len(records)}] "
                "skip completed"
            )
            continue

        if not snapshot_exists(
            protocol,
            index,
        ):
            missing_skipped += 1

            print(
                f"[{position}/"
                f"{len(records)}] "
                "snapshot missing"
            )

            continue

        snapshot = load_snapshot(
            protocol,
            index,
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        (
            gold_label,
            required_reannotation,
        ) = get_gold_metadata(
            index=index,
            dataset_by_index=(
                dataset_by_index
            ),
            raw_dataset=raw_dataset,
        )

        if not evidence:
            record = build_record(
                manifest_record=(
                    manifest_record
                ),
                snapshot=snapshot,
                gold_label=gold_label,
                required_reannotation=(
                    required_reannotation
                ),
                analysis=None,
                latency_ms=0,
            )

            records_by_index[
                index
            ] = record

            zero_evidence_completed += 1

            save_progress(
                records=records,
                records_by_index=(
                    records_by_index
                ),
                result_path=result_path,
            )

            print(
                f"[{position}/"
                f"{len(records)}] "
                "zero evidence -> NEI"
            )

            continue

        raw_path = (
            get_raw_result_path(
                protocol,
                index,
            )
        )

        if raw_path.exists():
            analysis = load_json(
                raw_path
            )

            record = build_record(
                manifest_record=(
                    manifest_record
                ),
                snapshot=snapshot,
                gold_label=gold_label,
                required_reannotation=(
                    required_reannotation
                ),
                analysis=analysis,
                latency_ms=0,
            )

            records_by_index[
                index
            ] = record

            recovered += 1

            save_progress(
                records=records,
                records_by_index=(
                    records_by_index
                ),
                result_path=result_path,
            )

            print(
                f"[{position}/"
                f"{len(records)}] "
                "recovered cached raw "
                "result"
            )

            continue

        if attempts >= max_attempts:
            llm_cap_skipped += 1

            print(
                f"[{position}/"
                f"{len(records)}] "
                "new-call cap reached"
            )

            continue

        attempts += 1

        start = time.perf_counter()

        try:
            analysis = (
                analyze_evidence_with_llm(
                    claim=snapshot[
                        "claim"
                    ],
                    evidence=evidence,
                )
            )
        except Exception as exc:
            print(
                f"[{position}/"
                f"{len(records)}] "
                "LLM ERROR:",
                type(exc).__name__,
            )

            print(
                "RUN STOPPED AFTER "
                "LLM ERROR."
            )

            stopped_on_error = True
            break

        latency_ms = round(
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        if not isinstance(
            analysis,
            dict,
        ):
            raise TypeError(
                "Evidence Analysis did "
                "not return a dict."
            )

        save_json(
            raw_path,
            analysis,
        )

        record = build_record(
            manifest_record=(
                manifest_record
            ),
            snapshot=snapshot,
            gold_label=gold_label,
            required_reannotation=(
                required_reannotation
            ),
            analysis=analysis,
            latency_ms=latency_ms,
        )

        records_by_index[
            index
        ] = record

        save_progress(
            records=records,
            records_by_index=(
                records_by_index
            ),
            result_path=result_path,
        )

        print(
            f"[{position}/"
            f"{len(records)}] "
            "completed"
        )

    ordered = save_progress(
        records=records,
        records_by_index=(
            records_by_index
        ),
        result_path=result_path,
    )

    completed = [
        item
        for item in ordered
        if "predicted_label"
        in item
    ]

    print()
    print(
        "===== VALIDATION "
        "PROGRESS ====="
    )
    print(
        "NEW LLM CALLS THIS RUN:",
        attempts,
    )
    print(
        "RAW RESULTS RECOVERED:",
        recovered,
    )
    print(
        "ZERO-EVIDENCE COMPLETED:",
        zero_evidence_completed,
    )
    print(
        "MISSING SNAPSHOTS SKIPPED:",
        missing_skipped,
    )
    print(
        "READY CASES SKIPPED BY "
        "LLM CAP:",
        llm_cap_skipped,
    )
    print(
        "STOPPED ON LLM ERROR:",
        stopped_on_error,
    )
    print(
        "COMPLETED:",
        len(completed),
    )
    print(
        "TOTAL:",
        len(records),
    )
    print(
        "OUTPUT:",
        result_path,
    )

    if (
        len(completed)
        != len(records)
    ):
        print(
            "METRICS WITHHELD UNTIL "
            "ALL 80 CASES COMPLETE."
        )
        return

    full_metrics = (
        compute_metrics(
            completed
        )
    )

    clean_records = [
        item
        for item in completed
        if not item.get(
            "required_reannotation",
            False,
        )
    ]

    flagged_records = [
        item
        for item in completed
        if item.get(
            "required_reannotation",
            False,
        )
    ]

    clean_metrics = (
        compute_metrics(
            clean_records
        )
    )

    summary = {
        "protocol": protocol,
        "sample_id": (
            "v2_validation_n80_"
            "seed4202"
        ),
        "primary_full_set": (
            full_metrics
        ),
        "predeclared_clean_subset": (
            clean_metrics
        ),
        "required_reannotation_count": (
            len(flagged_records)
        ),
        "clean_subset_definition": (
            "required_reannotation == "
            "False"
        ),
    }

    summary_path = (
        get_summary_path(
            protocol
        )
    )

    save_json(
        summary_path,
        summary,
    )

    print()
    print(
        "===== FROZEN VALIDATION "
        "COMPLETE ====="
    )
    print(
        "FULL N:",
        full_metrics["n"],
    )
    print(
        "FULL ACCURACY:",
        round(
            full_metrics[
                "accuracy"
            ],
            4,
        ),
    )
    print(
        "FULL MACRO F1:",
        round(
            full_metrics[
                "macro_f1"
            ],
            4,
        ),
    )
    print(
        "REANNOTATION FLAGGED:",
        len(flagged_records),
    )
    print(
        "CLEAN N:",
        clean_metrics["n"],
    )
    print(
        "CLEAN ACCURACY:",
        round(
            clean_metrics[
                "accuracy"
            ],
            4,
        ),
    )
    print(
        "CLEAN MACRO F1:",
        round(
            clean_metrics[
                "macro_f1"
            ],
            4,
        ),
    )
    print(
        "SUMMARY:",
        summary_path,
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--protocol",
        choices=[
            "averitec_standard",
            "trustnet_historical_cutoff",
        ],
        default="averitec_standard",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    parser.add_argument(
        "--confirm-live",
        action="store_true",
    )

    args = parser.parse_args()

    _, records = (
        load_validation_records()
    )

    verify_development_disjoint(
        records
    )

    (
        dataset_by_index,
        raw_dataset,
    ) = load_dataset_metadata()

    validate_against_dataset(
        records,
        dataset_by_index,
    )

    if args.dry_run:
        dry_run(
            records=records,
            protocol=args.protocol,
            max_attempts=(
                args.max_attempts
            ),
        )
        return

    run(
        records=records,
        dataset_by_index=(
            dataset_by_index
        ),
        raw_dataset=raw_dataset,
        protocol=args.protocol,
        max_attempts=(
            args.max_attempts
        ),
        confirm_live=(
            args.confirm_live
        ),
    )


if __name__ == "__main__":
    main()
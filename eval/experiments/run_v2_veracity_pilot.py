import argparse
import json
import time
from collections import Counter
from pathlib import Path

from backend.app.llm.evidence_analysis import (
    analyze_evidence_with_llm,
    load_evidence_prompt,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)
from eval.snapshots.averitec import (
    evidence_from_snapshot,
    load_averitec_snapshot,
)


SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_stratified/"
    "train/n12_seed4301/averitec_standard"
)

MANIFEST_PATH = (
    SNAPSHOT_ROOT
    / "manifest.json"
)

OUTPUT_PATH = Path(
    "eval/results/"
    "trustnet_v2_veracity_"
    "train_n12_seed4301.json"
)

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

    temporary = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
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


def load_manifest_records() -> list[dict]:
    manifest = load_json(
        MANIFEST_PATH
    )

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

    return examples


def select_balanced(
    records: list[dict],
    per_class: int,
) -> list[dict]:
    counts = Counter()
    selected = []

    for record in records:
        label = record.get(
            "label"
        )

        if label not in LABELS:
            continue

        if counts[label] >= per_class:
            continue

        selected.append(
            record
        )

        counts[label] += 1

    expected = (
        per_class
        * len(LABELS)
    )

    if len(selected) != expected:
        raise ValueError(
            "Could not create balanced "
            f"selection of {expected}. "
            f"Counts: {dict(counts)}"
        )

    return selected


def load_snapshot_for_record(
    record: dict,
) -> dict:
    index = record[
        "index"
    ]

    return load_averitec_snapshot(
        SNAPSHOT_ROOT
        / f"{index}.json"
    )


def approximate_prompt_chars(
    claim: str,
    evidence,
) -> int:
    system_prompt = (
        load_evidence_prompt()
    )

    parts = []

    for item in evidence:
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=claim,
                max_chars=(
                    DEFAULT_PASSAGE_MAX_CHARS
                ),
            )
        )

        parts.append(
            f"""
        Title: {item.title}
        URL: {item.url}
        Source: {item.source}
        Quality: {item.quality}
        Origin: {item.evidence_origin}
        Evidence:
        {excerpt}
        """
        )

    evidence_text = (
        "\n\n".join(
            parts
        )
    )

    user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}
"""

    return (
        len(system_prompt)
        + len(user_prompt)
    )


def dry_run(
    selected: list[dict],
):
    total_chars = 0
    evidence_count = 0
    zero_evidence = 0
    max_chars = 0

    counts = Counter()

    for record in selected:
        snapshot = (
            load_snapshot_for_record(
                record
            )
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        claim = snapshot[
            "claim"
        ]

        counts[
            record["label"]
        ] += 1

        evidence_count += len(
            evidence
        )

        if not evidence:
            zero_evidence += 1

        chars = (
            approximate_prompt_chars(
                claim=claim,
                evidence=evidence,
            )
        )

        total_chars += chars

        max_chars = max(
            max_chars,
            chars,
        )

    print(
        "===== V2 VERACITY DRY RUN ====="
    )
    print(
        "CASES:",
        len(selected),
    )
    print(
        "CLASS COUNTS:",
        dict(counts),
    )
    print(
        "EVIDENCE ITEMS:",
        evidence_count,
    )
    print(
        "ZERO-EVIDENCE CASES:",
        zero_evidence,
    )
    print(
        "TOTAL PROMPT CHARS:",
        total_chars,
    )
    print(
        "AVG PROMPT CHARS:",
        round(
            total_chars
            / len(selected),
            1,
        ),
    )
    print(
        "MAX PROMPT CHARS:",
        max_chars,
    )
    print(
        "APPROX INPUT TOKENS:",
        round(
            total_chars
            / 4
        ),
    )
    print(
        "OPENAI CALLS:",
        0,
    )
    print(
        "TAVILY CALLS:",
        0,
    )


def load_existing() -> list[dict]:
    if not OUTPUT_PATH.exists():
        return []

    data = load_json(
        OUTPUT_PATH
    )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "Existing result file must "
            "contain a JSON list."
        )

    return data


def run(
    selected: list[dict],
    max_attempts: int,
):
    selected_indices = {
        item["index"]
        for item in selected
    }

    existing = (
        load_existing()
    )

    records_by_index = {
        item["index"]: item
        for item in existing
        if (
            isinstance(
                item,
                dict,
            )
            and item.get("index")
            in selected_indices
        )
    }

    attempts = 0

    for number, manifest_record in enumerate(
        selected,
        start=1,
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
                f"[{number}/"
                f"{len(selected)}] "
                "SKIP completed"
            )
            continue

        snapshot = (
            load_snapshot_for_record(
                manifest_record
            )
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        claim = snapshot[
            "claim"
        ]

        gold_label = (
            manifest_record[
                "label"
            ]
        )

        if not evidence:
            prediction = (
                "Not Enough Evidence"
            )

            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "trustnet_v2_"
                    "evidence_only"
                ),
                "sample_id": (
                    "n12_seed4301"
                ),
                "index": index,
                "claim": claim,
                "gold_label": (
                    gold_label
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
                    "insufficient"
                ),
                "llm_evidence_direction": (
                    None
                ),
                "direction_was_resolved": (
                    False
                ),
                "subclaim_assessments": [],
                "evidence_assessments": [],
                "uncertainty": {
                    "gate_triggered": False,
                    "level": "none",
                    "reason": (
                        "Evidence-only v2 "
                        "pilot."
                    ),
                },
                "analysis_latency_ms": (
                    0
                ),
                "llm_timings": {},
                "sources": [],
            }

            records_by_index[
                index
            ] = record

            print(
                f"[{number}/"
                f"{len(selected)}] "
                "zero evidence -> NEI"
            )

            continue

        if attempts >= max_attempts:
            print(
                "LLM attempt limit reached."
            )
            break

        attempts += 1

        start = (
            time.perf_counter()
        )

        try:
            analysis = (
                analyze_evidence_with_llm(
                    claim=claim,
                    evidence=evidence,
                )
            )

            latency_ms = round(
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            )

            direction = (
                analysis[
                    "overall_evidence_direction"
                ]
            )

            prediction = (
                DIRECTION_TO_LABEL[
                    direction
                ]
            )

            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "trustnet_v2_"
                    "evidence_only"
                ),
                "sample_id": (
                    "n12_seed4301"
                ),
                "index": index,
                "claim": claim,
                "gold_label": (
                    gold_label
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
                    analysis.get(
                        "llm_overall_"
                        "evidence_direction"
                    )
                ),
                "direction_was_resolved": (
                    analysis.get(
                        "direction_was_resolved",
                        False,
                    )
                ),
                "subclaim_assessments": (
                    analysis.get(
                        "subclaim_assessments",
                        [],
                    )
                ),
                "evidence_assessments": (
                    analysis.get(
                        "evidence_assessments",
                        [],
                    )
                ),
                "uncertainty": {
                    "gate_triggered": False,
                    "level": "none",
                    "reason": (
                        "Evidence-only v2 "
                        "pilot."
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

            print(
                f"[{number}/"
                f"{len(selected)}] "
                "completed"
            )

        except Exception as exc:
            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "trustnet_v2_"
                    "evidence_only"
                ),
                "sample_id": (
                    "n12_seed4301"
                ),
                "index": index,
                "claim": claim,
                "gold_label": (
                    gold_label
                ),
                "error": str(
                    exc
                ),
            }

            print(
                f"[{number}/"
                f"{len(selected)}] "
                f"ERROR: {exc}"
            )

        records_by_index[
            index
        ] = record

        ordered = [
            records_by_index[
                item["index"]
            ]
            for item in selected
            if item["index"]
            in records_by_index
        ]

        save_json(
            OUTPUT_PATH,
            ordered,
        )

    ordered = [
        records_by_index[
            item["index"]
        ]
        for item in selected
        if item["index"]
        in records_by_index
    ]

    save_json(
        OUTPUT_PATH,
        ordered,
    )

    completed = [
        item
        for item in ordered
        if "predicted_label"
        in item
    ]

    errors = [
        item
        for item in ordered
        if "error" in item
    ]

    correct = sum(
        item["predicted_label"]
        == item["gold_label"]
        for item in completed
    )

    print()
    print(
        "===== V2 VERACITY PILOT ====="
    )
    print(
        "SELECTED:",
        len(selected),
    )
    print(
        "LLM ATTEMPTS THIS RUN:",
        attempts,
    )
    print(
        "COMPLETED:",
        len(completed),
    )
    print(
        "ERRORS:",
        len(errors),
    )
    print(
        "CORRECT:",
        correct,
    )

    if completed:
        print(
            "ACCURACY:",
            round(
                correct
                / len(completed),
                3,
            ),
        )

    print(
        "OUTPUT:",
        OUTPUT_PATH,
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if not (
        1 <= args.per_class <= 3
    ):
        raise ValueError(
            "--per-class must be "
            "between 1 and 3."
        )

    if args.max_attempts < 1:
        raise ValueError(
            "--max-attempts must "
            "be at least 1."
        )

    records = (
        load_manifest_records()
    )

    selected = (
        select_balanced(
            records=records,
            per_class=(
                args.per_class
            ),
        )
    )

    if args.dry_run:
        dry_run(
            selected
        )
        return

    run(
        selected=selected,
        max_attempts=(
            args.max_attempts
        ),
    )


if __name__ == "__main__":
    main()
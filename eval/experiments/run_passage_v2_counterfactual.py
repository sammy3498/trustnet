import argparse
import json
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


SOURCE_RESULTS_PATH = Path(
    "eval/results/"
    "averitec_stratified_train_"
    "averitec_standard_n20_seed3001.json"
)

OUTPUT_PATH = Path(
    "eval/results/"
    "passage_v2_counterfactual_"
    "seed3001_n8.json"
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


def select_balanced_records(
    records: list[dict],
    per_class: int,
) -> list[dict]:
    selected = []
    counts = Counter()

    for record in records:
        label = record.get(
            "gold_label"
        )

        if label not in LABELS:
            continue

        if counts[label] >= per_class:
            continue

        selected.append(
            record
        )

        counts[label] += 1

        if all(
            counts[label] >= per_class
            for label in LABELS
        ):
            break

    expected = (
        per_class
        * len(LABELS)
    )

    if len(selected) != expected:
        raise ValueError(
            "Could not construct balanced "
            f"sample of {expected} cases. "
            f"Counts: {dict(counts)}"
        )

    return selected


def build_prompt_character_count(
    claim: str,
    evidence,
) -> int:
    system_prompt = (
        load_evidence_prompt()
    )

    evidence_parts = []

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

        evidence_parts.append(
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
            evidence_parts
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
            "Existing output must "
            "contain a JSON list."
        )

    return data


def dry_run(
    selected: list[dict],
):
    total_chars = 0
    largest_chars = 0
    evidence_items = 0
    zero_evidence = 0

    for record in selected:
        snapshot_path = Path(
            record[
                "snapshot_path"
            ]
        )

        snapshot = (
            load_averitec_snapshot(
                snapshot_path
            )
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        evidence_items += len(
            evidence
        )

        if not evidence:
            zero_evidence += 1

        chars = (
            build_prompt_character_count(
                claim=record["claim"],
                evidence=evidence,
            )
        )

        total_chars += chars

        largest_chars = max(
            largest_chars,
            chars,
        )

    approximate_tokens = (
        total_chars // 4
    )

    print(
        "===== PASSAGE V2 DRY RUN ====="
    )
    print(
        f"CASES: {len(selected)}"
    )
    print(
        "CLASS COUNTS:",
        dict(
            Counter(
                item["gold_label"]
                for item in selected
            )
        ),
    )
    print(
        "EVIDENCE ITEMS:",
        evidence_items,
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
        largest_chars,
    )
    print(
        "APPROX INPUT TOKENS:",
        approximate_tokens,
    )
    print(
        "OPENAI CALLS:",
        0,
    )
    print(
        "TAVILY CALLS:",
        0,
    )


def run_counterfactual(
    selected: list[dict],
    max_attempts: int,
):
    existing = (
        load_existing()
    )

    records_by_index = {
        item["index"]: item
        for item in existing
        if isinstance(
            item,
            dict,
        )
        and isinstance(
            item.get("index"),
            int,
        )
    }

    attempts = 0

    for number, record in enumerate(
        selected,
        start=1,
    ):
        index = record[
            "index"
        ]

        old = (
            records_by_index.get(
                index
            )
        )

        if (
            old is not None
            and "v2_predicted_label"
            in old
        ):
            print(
                f"[{number}/"
                f"{len(selected)}] "
                "SKIP completed"
            )
            continue

        if attempts >= max_attempts:
            print(
                "LLM attempt limit reached."
            )
            break

        snapshot = (
            load_averitec_snapshot(
                Path(
                    record[
                        "snapshot_path"
                    ]
                )
            )
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        if not evidence:
            result = {
                "index": index,
                "gold_label": (
                    record[
                        "gold_label"
                    ]
                ),
                "v1_predicted_label": (
                    record[
                        "predicted_label"
                    ]
                ),
                "v2_predicted_label": (
                    "Not Enough Evidence"
                ),
                "v2_direction": (
                    "insufficient"
                ),
                "llm_call_used": False,
            }

            records_by_index[
                index
            ] = result

            print(
                f"[{number}/"
                f"{len(selected)}] "
                "zero evidence -> NEI"
            )

            continue

        attempts += 1

        try:
            analysis = (
                analyze_evidence_with_llm(
                    claim=(
                        record[
                            "claim"
                        ]
                    ),
                    evidence=evidence,
                )
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

            result = {
                "index": index,
                "gold_label": (
                    record[
                        "gold_label"
                    ]
                ),
                "v1_predicted_label": (
                    record[
                        "predicted_label"
                    ]
                ),
                "v2_predicted_label": (
                    prediction
                ),
                "v2_direction": (
                    direction
                ),
                "llm_call_used": True,
            }

            print(
                f"[{number}/"
                f"{len(selected)}] "
                "completed"
            )

        except Exception as exc:
            result = {
                "index": index,
                "gold_label": (
                    record[
                        "gold_label"
                    ]
                ),
                "v1_predicted_label": (
                    record[
                        "predicted_label"
                    ]
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
        ] = result

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
        if "v2_predicted_label"
        in item
    ]

    errors = [
        item
        for item in ordered
        if "error" in item
    ]

    v1_correct = sum(
        item[
            "v1_predicted_label"
        ]
        == item[
            "gold_label"
        ]
        for item in completed
    )

    v2_correct = sum(
        item[
            "v2_predicted_label"
        ]
        == item[
            "gold_label"
        ]
        for item in completed
    )

    changed = sum(
        item[
            "v1_predicted_label"
        ]
        != item[
            "v2_predicted_label"
        ]
        for item in completed
    )

    print()
    print(
        "===== PASSAGE V2 "
        "COUNTERFACTUAL ====="
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
        "V1 CORRECT:",
        v1_correct,
    )
    print(
        "V2 CORRECT:",
        v2_correct,
    )
    print(
        "PREDICTIONS CHANGED:",
        changed,
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
        default=2,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    if args.per_class < 1:
        raise ValueError(
            "--per-class must "
            "be at least 1."
        )

    if args.max_attempts < 1:
        raise ValueError(
            "--max-attempts must "
            "be at least 1."
        )

    source_records = (
        load_json(
            SOURCE_RESULTS_PATH
        )
    )

    selected = (
        select_balanced_records(
            records=source_records,
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

    run_counterfactual(
        selected=selected,
        max_attempts=(
            args.max_attempts
        ),
    )


if __name__ == "__main__":
    main()
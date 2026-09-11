import argparse
import json
import time
from pathlib import Path

from backend.app.llm.client import get_llm_response
from eval.loaders.averitec import load_averitec


DATASET_PATH = Path(
    "eval/datasets/averitec/train.json"
)

DEFAULT_MANIFEST_PATH = Path(
    "eval/snapshots/averitec_stratified/"
    "train_final_holdout.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "eval/results/"
    "averitec_plain_llm_baseline_"
    "train_n100_seed2026.json"
)

PROMPT_PATH = Path(
    "prompts/"
    "averitec_plain_llm_baseline_v1.txt"
)

VALID_LABELS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}

BASELINE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": [
                "Supported",
                "Refuted",
                "Not Enough Evidence",
                "Conflicting Evidence/Cherrypicking",
            ],
        },
    },
    "required": [
        "label",
    ],
    "additionalProperties": False,
}


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def load_manifest_indices(
    path: Path,
) -> list[int]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    examples = data.get("examples")

    if not isinstance(examples, list):
        raise ValueError(
            f"{path} does not contain "
            "an examples list."
        )

    indices = []

    for item in examples:
        index = item.get("index")

        if not isinstance(index, int):
            raise ValueError(
                "Invalid example index "
                f"in {path}."
            )

        indices.append(index)

    if not indices:
        raise ValueError(
            f"No examples found in {path}."
        )

    if len(set(indices)) != len(indices):
        raise ValueError(
            "Duplicate indices found "
            f"in {path}."
        )

    return indices


def load_result_file(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(
            f"{path} must contain "
            "a JSON list."
        )

    return data


def save_results(
    path: Path,
    results: list[dict],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.with_suffix(
            path.suffix + ".tmp"
        )
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    temporary_path.replace(path)


def classify_claim(
    claim: str,
    system_prompt: str,
) -> tuple[str, int]:
    start = time.perf_counter()

    raw_response = get_llm_response(
        system_prompt=system_prompt,
        user_prompt=(
            f"Claim:\n{claim}"
        ),
        response_schema=(
            BASELINE_RESPONSE_SCHEMA
        ),
        response_schema_name=(
            "averitec_plain_llm_baseline"
        ),
    )

    latency_ms = round(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

    try:
        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid "
            f"JSON:\n{raw_response}"
        ) from exc

    label = data.get("label")

    if label not in VALID_LABELS:
        raise ValueError(
            "LLM returned invalid "
            f"label: {label}"
        )

    return label, latency_ms


def run(
    manifest_path: Path,
    output_path: Path,
    cache_from: Path | None = None,
    overwrite: bool = False,
    max_attempts: int | None = None,
):
    if (
        max_attempts is not None
        and max_attempts < 1
    ):
        raise ValueError(
            "--max-attempts must be "
            "at least 1."
        )

    examples = load_averitec(
        DATASET_PATH,
        skip_invalid_claims=True,
    )

    examples_by_index = {
        example.index: example
        for example in examples
    }

    selected_indices = (
        load_manifest_indices(
            manifest_path
        )
    )

    selected = []

    for index in selected_indices:
        example = examples_by_index.get(
            index
        )

        if example is None:
            raise ValueError(
                "Selected index is missing "
                f"from valid dataset: {index}"
            )

        selected.append(example)

    selected_index_set = set(
        selected_indices
    )

    system_prompt = load_prompt()

    records_by_index = {}

    if (
        cache_from is not None
        and cache_from.exists()
    ):
        cached_records = (
            load_result_file(
                cache_from
            )
        )

        for item in cached_records:
            if not isinstance(
                item,
                dict,
            ):
                continue

            index = item.get("index")

            if (
                index in selected_index_set
                and "predicted_label"
                in item
                and "operational_prediction"
                in item
            ):
                records_by_index[
                    index
                ] = item

    if not overwrite:
        existing_records = (
            load_result_file(
                output_path
            )
        )

        for item in existing_records:
            if not isinstance(
                item,
                dict,
            ):
                continue

            index = item.get("index")

            if index in selected_index_set:
                records_by_index[
                    index
                ] = item

    elif output_path.exists():
        output_path.unlink()

    reused_completed = sum(
        1
        for index in selected_indices
        if (
            index in records_by_index
            and "predicted_label"
            in records_by_index[index]
            and "operational_prediction"
            in records_by_index[index]
        )
    )

    print(
        "BASELINE: plain_llm"
    )
    print(
        "DATASET: AVeriTeC train"
    )
    print(
        f"MANIFEST: {manifest_path}"
    )
    print(
        f"TOTAL SELECTED: {len(selected)}"
    )
    print(
        "REUSED COMPLETED: "
        f"{reused_completed}"
    )
    print(
        "MAX API ATTEMPTS THIS RUN: "
        f"{max_attempts}"
    )
    print(
        f"OUTPUT: {output_path}"
    )

    attempts_this_run = 0

    for number, example in enumerate(
        selected,
        start=1,
    ):
        existing = records_by_index.get(
            example.index
        )

        if (
            existing is not None
            and "predicted_label"
            in existing
            and "operational_prediction"
            in existing
        ):
            print(
                f"[{number}/{len(selected)}] "
                "SKIP completed"
            )
            continue

        if (
            max_attempts is not None
            and attempts_this_run
            >= max_attempts
        ):
            print(
                "API attempt limit reached."
            )
            break

        attempts_this_run += 1

        try:
            (
                prediction,
                latency_ms,
            ) = classify_claim(
                claim=example.claim,
                system_prompt=system_prompt,
            )

            correct = (
                prediction
                == example.label
            )

            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "plain_llm_baseline"
                ),
                "sample_id": (
                    manifest_path.stem
                ),
                "baseline_type": (
                    "plain_llm"
                ),
                "index": example.index,
                "claim": example.claim,
                "gold_label": (
                    example.label
                ),
                "predicted_label": (
                    prediction
                ),
                "operational_prediction": (
                    prediction
                ),
                "correct": correct,
                "operational_correct": (
                    correct
                ),
                "overall_score": None,
                "credibility_label": None,
                "evidence_direction": None,
                "llm_evidence_direction": None,
                "direction_was_resolved": (
                    False
                ),
                "subclaim_assessments": [],
                "evidence_assessments": [],
                "uncertainty": {
                    "gate_triggered": False,
                    "level": "none",
                    "reason": (
                        "Plain LLM baseline has "
                        "no TrustNet uncertainty "
                        "gate."
                    ),
                },
                "factors": {},
                "llm_timings": {
                    "baseline_llm_ms": (
                        latency_ms
                    ),
                },
                "analysis_latency_ms": (
                    latency_ms
                ),
                "context": None,
                "retrieval_end_date": None,
                "search_query": None,
                "verification_queries": [],
                "retrieval_config": None,
                "snapshot_path": None,
                "snapshot_sha256": None,
                "sources": [],
            }

            print(
                f"[{number}/{len(selected)}] "
                "completed"
            )

        except Exception as exc:
            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "plain_llm_baseline"
                ),
                "sample_id": (
                    manifest_path.stem
                ),
                "baseline_type": (
                    "plain_llm"
                ),
                "index": example.index,
                "claim": example.claim,
                "gold_label": (
                    example.label
                ),
                "error": str(exc),
            }

            print(
                f"[{number}/{len(selected)}] "
                f"ERROR: {exc}"
            )

        records_by_index[
            example.index
        ] = record

        ordered_results = [
            records_by_index[
                item.index
            ]
            for item in selected
            if item.index
            in records_by_index
        ]

        save_results(
            output_path,
            ordered_results,
        )

    results = [
        records_by_index[
            example.index
        ]
        for example in selected
        if example.index
        in records_by_index
    ]

    save_results(
        output_path,
        results,
    )

    completed = [
        item
        for item in results
        if (
            "predicted_label"
            in item
            and "operational_prediction"
            in item
        )
    ]

    errors = [
        item
        for item in results
        if "error" in item
    ]

    correct = sum(
        bool(
            item.get("correct")
        )
        for item in completed
    )

    print()
    print(
        "===== PLAIN LLM BASELINE ====="
    )
    print(
        "API attempts this run: "
        f"{attempts_this_run}"
    )
    print(
        "Completed predictions stored: "
        f"{len(completed)}/"
        f"{len(selected)}"
    )
    print(
        f"Stored errors: {len(errors)}"
    )

    if completed:
        print(
            "Correct among completed: "
            f"{correct}/"
            f"{len(completed)}"
        )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=(
            DEFAULT_MANIFEST_PATH
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DEFAULT_OUTPUT_PATH
        ),
    )

    parser.add_argument(
        "--cache-from",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of paid LLM "
            "attempts allowed in this run."
        ),
    )

    args = parser.parse_args()

    run(
        manifest_path=args.manifest,
        output_path=args.output,
        cache_from=args.cache_from,
        overwrite=args.overwrite,
        max_attempts=args.max_attempts,
    )
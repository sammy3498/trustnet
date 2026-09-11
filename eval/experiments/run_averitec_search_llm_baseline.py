import argparse
import json
import math
import time
from pathlib import Path

from backend.app.llm.client import get_llm_response
from eval.loaders.averitec import load_averitec


DATASET_PATH = Path(
    "eval/datasets/averitec/train.json"
)

DEFAULT_MANIFEST_PATH = Path(
    "eval/snapshots/averitec_stratified/"
    "plain_llm_baseline_n20.json"
)

DEFAULT_TRUSTNET_RESULTS_PATH = Path(
    "eval/results/"
    "averitec_trustnet_same_n20_balanced.json"
)

DEFAULT_OUTPUT_PATH = Path(
    "eval/results/"
    "averitec_search_llm_baseline_"
    "train_n20_balanced.json"
)

PROMPT_PATH = Path(
    "prompts/"
    "averitec_search_llm_baseline_v1.txt"
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


def load_json(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def load_manifest_indices(
    path: Path,
) -> list[int]:
    data = load_json(path)

    examples = data.get("examples")

    if not isinstance(
        examples,
        list,
    ):
        raise ValueError(
            f"{path} does not contain "
            "an examples list."
        )

    indices = []

    for item in examples:
        index = item.get("index")

        if not isinstance(
            index,
            int,
        ):
            raise ValueError(
                "Invalid example index "
                f"in {path}."
            )

        indices.append(index)

    if not indices:
        raise ValueError(
            f"No examples found in {path}."
        )

    if len(set(indices)) != len(
        indices
    ):
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

    data = load_json(path)

    if not isinstance(
        data,
        list,
    ):
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


def load_trustnet_records(
    path: Path,
) -> dict[int, dict]:
    records = load_result_file(path)

    result = {}

    for record in records:
        if not isinstance(
            record,
            dict,
        ):
            continue

        index = record.get("index")

        if isinstance(
            index,
            int,
        ):
            result[index] = record

    return result


def load_snapshot(
    record: dict,
) -> dict:
    snapshot_path = record.get(
        "snapshot_path"
    )

    if not snapshot_path:
        raise ValueError(
            "TrustNet record does not "
            "contain snapshot_path."
        )

    path = Path(snapshot_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot not found: {path}"
        )

    snapshot = load_json(path)

    if not isinstance(
        snapshot,
        dict,
    ):
        raise ValueError(
            f"Invalid snapshot: {path}"
        )

    return snapshot


def format_evidence(
    snapshot: dict,
) -> tuple[str, list[dict]]:
    evidence = snapshot.get(
        "evidence"
    )

    if not isinstance(
        evidence,
        list,
    ):
        raise ValueError(
            "Snapshot evidence must "
            "be a list."
        )

    blocks = []
    source_metadata = []

    for number, item in enumerate(
        evidence,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            continue

        title = (
            item.get("title")
            or "Unknown"
        )

        source = (
            item.get("source")
            or "Unknown"
        )

        url = (
            item.get("url")
            or ""
        )

        published_date = (
            item.get("published_date")
            or "Unknown"
        )

        origin = (
            item.get("evidence_origin")
            or "unknown"
        )

        text = (
            item.get("text")
            or ""
        ).strip()

        block = (
            f"EVIDENCE {number}\n"
            f"Title: {title}\n"
            f"Source: {source}\n"
            f"Published date: "
            f"{published_date}\n"
            f"Origin: {origin}\n"
            f"URL: {url}\n"
            f"Text:\n{text}"
        )

        blocks.append(block)

        source_metadata.append(
            {
                "title": title,
                "source": source,
                "url": url,
                "published_date": (
                    None
                    if published_date
                    == "Unknown"
                    else published_date
                ),
                "origin": origin,
            }
        )

    return (
        "\n\n---\n\n".join(
            blocks
        ),
        source_metadata,
    )


def build_user_prompt(
    claim: str,
    evidence_text: str,
) -> str:
    return (
        "CLAIM:\n"
        f"{claim}\n\n"
        "RETRIEVED EVIDENCE:\n"
        f"{evidence_text}"
    )


def classify_claim(
    claim: str,
    evidence_text: str,
    system_prompt: str,
) -> tuple[str, int]:
    user_prompt = (
        build_user_prompt(
            claim=claim,
            evidence_text=evidence_text,
        )
    )

    start = time.perf_counter()

    raw_response = get_llm_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_schema=(
            BASELINE_RESPONSE_SCHEMA
        ),
        response_schema_name=(
            "averitec_search_llm_baseline"
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


def estimate_inputs(
    selected,
    trustnet_records,
    system_prompt: str,
):
    total_evidence = 0
    total_chars = 0
    largest_chars = 0

    for example in selected:
        record = trustnet_records.get(
            example.index
        )

        if record is None:
            raise ValueError(
                "Missing TrustNet record "
                f"for index {example.index}."
            )

        snapshot = load_snapshot(
            record
        )

        evidence_text, sources = (
            format_evidence(
                snapshot
            )
        )

        user_prompt = (
            build_user_prompt(
                claim=example.claim,
                evidence_text=evidence_text,
            )
        )

        chars = (
            len(system_prompt)
            + len(user_prompt)
        )

        total_evidence += len(
            sources
        )

        total_chars += chars

        largest_chars = max(
            largest_chars,
            chars,
        )

    average_chars = (
        total_chars
        / len(selected)
    )

    approximate_total_tokens = (
        math.ceil(
            total_chars / 4
        )
    )

    approximate_average_tokens = (
        math.ceil(
            average_chars / 4
        )
    )

    approximate_largest_tokens = (
        math.ceil(
            largest_chars / 4
        )
    )

    print(
        "===== DRY RUN ====="
    )
    print(
        f"CASES: {len(selected)}"
    )
    print(
        "TOTAL CACHED EVIDENCE ITEMS: "
        f"{total_evidence}"
    )
    print(
        "TOTAL INPUT CHARACTERS: "
        f"{total_chars}"
    )
    print(
        "AVERAGE INPUT CHARACTERS: "
        f"{average_chars:.1f}"
    )
    print(
        "LARGEST INPUT CHARACTERS: "
        f"{largest_chars}"
    )
    print(
        "APPROX TOTAL INPUT TOKENS: "
        f"{approximate_total_tokens}"
    )
    print(
        "APPROX AVG INPUT TOKENS/CASE: "
        f"{approximate_average_tokens}"
    )
    print(
        "APPROX MAX INPUT TOKENS/CASE: "
        f"{approximate_largest_tokens}"
    )
    print(
        "RETRIEVAL API CALLS: 0"
    )
    print(
        "LLM API CALLS IN DRY RUN: 0"
    )


def run(
    manifest_path: Path,
    trustnet_results_path: Path,
    output_path: Path,
    overwrite: bool = False,
    max_attempts: int | None = None,
    dry_run: bool = False,
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

    trustnet_records = (
        load_trustnet_records(
            trustnet_results_path
        )
    )

    missing = [
        index
        for index in selected_indices
        if index
        not in trustnet_records
    ]

    if missing:
        raise ValueError(
            "TrustNet results are missing "
            f"{len(missing)} selected cases."
        )

    system_prompt = load_prompt()

    if dry_run:
        estimate_inputs(
            selected=selected,
            trustnet_records=(
                trustnet_records
            ),
            system_prompt=(
                system_prompt
            ),
        )
        return

    if overwrite:
        existing_records = []

        if output_path.exists():
            output_path.unlink()

    else:
        existing_records = (
            load_result_file(
                output_path
            )
        )

    selected_index_set = set(
        selected_indices
    )

    records_by_index = {}

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

    already_completed = sum(
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
        "BASELINE: search_llm"
    )
    print(
        "RETRIEVAL: cached TrustNet "
        "retrieval snapshots"
    )
    print(
        "NEW RETRIEVAL CALLS: 0"
    )
    print(
        f"TOTAL SELECTED: {len(selected)}"
    )
    print(
        "ALREADY COMPLETED: "
        f"{already_completed}"
    )
    print(
        "MAX LLM ATTEMPTS THIS RUN: "
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
                "LLM attempt limit reached."
            )
            break

        trustnet_record = (
            trustnet_records[
                example.index
            ]
        )

        snapshot = load_snapshot(
            trustnet_record
        )

        (
            evidence_text,
            source_metadata,
        ) = format_evidence(
            snapshot
        )

        attempts_this_run += 1

        try:
            (
                prediction,
                latency_ms,
            ) = classify_claim(
                claim=example.claim,
                evidence_text=(
                    evidence_text
                ),
                system_prompt=(
                    system_prompt
                ),
            )

            correct = (
                prediction
                == example.label
            )

            record = {
                "dataset": "AVeriTeC",
                "dataset_split": "train",
                "evaluation_protocol": (
                    "search_llm_baseline_"
                    "cached_retrieval"
                ),
                "sample_id": (
                    manifest_path.stem
                ),
                "baseline_type": (
                    "search_llm"
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
                        "Search + LLM baseline "
                        "has no TrustNet "
                        "uncertainty gate."
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
                "retrieval_end_date": (
                    snapshot.get(
                        "retrieval_end_date"
                    )
                ),
                "search_query": None,
                "verification_queries": [],
                "retrieval_config": (
                    snapshot.get(
                        "retrieval_config"
                    )
                ),
                "snapshot_path": (
                    trustnet_record.get(
                        "snapshot_path"
                    )
                ),
                "snapshot_sha256": (
                    trustnet_record.get(
                        "snapshot_sha256"
                    )
                ),
                "sources": (
                    source_metadata
                ),
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
                    "search_llm_baseline_"
                    "cached_retrieval"
                ),
                "sample_id": (
                    manifest_path.stem
                ),
                "baseline_type": (
                    "search_llm"
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
        "===== SEARCH + LLM BASELINE ====="
    )
    print(
        "New retrieval calls: 0"
    )
    print(
        "LLM attempts this run: "
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
        "--trustnet-results",
        type=Path,
        default=(
            DEFAULT_TRUSTNET_RESULTS_PATH
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
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help=(
            "Maximum number of paid "
            "LLM attempts allowed "
            "in this run."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Inspect cached evidence "
            "and estimate input size "
            "without making API calls."
        ),
    )

    args = parser.parse_args()

    run(
        manifest_path=args.manifest,
        trustnet_results_path=(
            args.trustnet_results
        ),
        output_path=args.output,
        overwrite=args.overwrite,
        max_attempts=(
            args.max_attempts
        ),
        dry_run=args.dry_run,
    )
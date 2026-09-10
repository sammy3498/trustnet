import argparse
import json
import time
from pathlib import Path

from backend.app.llm.pipeline import (
    analyze_claim_with_llm,
)
from eval.loaders.averitec import (
    load_averitec,
)
from eval.loaders.context import (
    AVERITEC_STANDARD_PROTOCOL,
    EVALUATION_PROTOCOLS,
)
from eval.metrics.label_mapping import (
    trustnet_operational_to_averitec,
    trustnet_to_averitec,
)
from eval.sampling.averitec import (
    AVERITEC_LABEL_ORDER,
    DEFAULT_SAMPLE_SEED,
    select_stratified_examples,
)
from eval.snapshots.averitec import (
    evidence_from_snapshot,
    get_snapshot_path,
    load_averitec_snapshot,
    snapshot_sha256,
)


DATASET_PATH = Path(
    "eval/datasets/averitec/dev.json"
)

SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_stratified"
)

RESULTS_DIR = Path(
    "eval/results"
)

DEFAULT_PER_CLASS = 5


def get_sample_id(
    per_class: int,
    seed: int,
) -> str:
    sample_size = (
        per_class
        * len(
            AVERITEC_LABEL_ORDER
        )
    )

    return (
        f"n{sample_size}_seed{seed}"
    )


def get_sample_root(
    per_class: int,
    seed: int,
) -> Path:
    return (
        SNAPSHOT_ROOT
        / get_sample_id(
            per_class,
            seed,
        )
    )


def get_manifest_path(
    protocol: str,
    per_class: int,
    seed: int,
) -> Path:
    return (
        get_sample_root(
            per_class,
            seed,
        )
        / protocol
        / "manifest.json"
    )


def get_output_path(
    protocol: str,
    per_class: int,
    seed: int,
) -> Path:
    return (
        RESULTS_DIR
        / (
            "averitec_stratified_"
            f"{protocol}_"
            f"{get_sample_id(per_class, seed)}"
            ".json"
        )
    )


def build_unclear_result(
    reason: str,
) -> dict:
    return {
        "overall_score": None,
        "label": (
            "Insufficient Evidence"
        ),
        "factors": {},
        "flags": [],
        "llm_timings": {},
        "uncertainty": {
            "gate_triggered": True,
            "level": "high",
            "reason": reason,
        },
        "evidence_analysis": {
            "overall_evidence_direction": (
                "insufficient"
            ),
            "evidence_strength_score": 0,
            "summary": reason,
            "evidence_assessments": [],
            "subclaim_assessments": [],
            "llm_overall_evidence_direction": (
                "insufficient"
            ),
            "llm_evidence_strength_score": 0,
            "direction_was_resolved": False,
        },
    }


def load_manifest(
    protocol: str,
    per_class: int,
    seed: int,
) -> dict:
    path = get_manifest_path(
        protocol,
        per_class,
        seed,
    )

    if not path.exists():
        raise FileNotFoundError(
            "Missing stratified manifest: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def validate_manifest(
    manifest: dict,
    selected,
    protocol: str,
    per_class: int,
    seed: int,
):
    if (
        manifest.get(
            "evaluation_protocol"
        )
        != protocol
    ):
        raise ValueError(
            "Manifest protocol does not "
            "match requested protocol."
        )

    if (
        manifest.get(
            "per_class"
        )
        != per_class
    ):
        raise ValueError(
            "Manifest per_class does not "
            "match requested sample."
        )

    if (
        manifest.get(
            "seed"
        )
        != seed
    ):
        raise ValueError(
            "Manifest seed does not "
            "match requested seed."
        )

    if (
        manifest.get(
            "sample_size"
        )
        != len(selected)
    ):
        raise ValueError(
            "Manifest sample size does "
            "not match selected sample."
        )

    expected_sample_id = (
        get_sample_id(
            per_class,
            seed,
        )
    )

    if (
        manifest.get(
            "sample_id"
        )
        != expected_sample_id
    ):
        raise ValueError(
            "Manifest sample ID does "
            "not match requested sample."
        )

    manifest_examples = (
        manifest.get(
            "examples",
            [],
        )
    )

    expected_examples = [
        {
            "index": example.index,
            "label": example.label,
            "claim": example.claim,
        }
        for example in selected
    ]

    if (
        manifest_examples
        != expected_examples
    ):
        raise ValueError(
            "Manifest examples do not "
            "match deterministic sample."
        )


def validate_snapshot(
    snapshot: dict,
    example,
    protocol: str,
):
    if (
        snapshot[
            "example_index"
        ]
        != example.index
    ):
        raise ValueError(
            "Snapshot index does not "
            "match dataset example."
        )

    if (
        snapshot[
            "claim"
        ]
        != example.claim
    ):
        raise ValueError(
            "Snapshot claim does not "
            "match dataset example."
        )

    if (
        snapshot.get(
            "gold_label"
        )
        != example.label
    ):
        raise ValueError(
            "Snapshot gold label does "
            "not match dataset example."
        )

    if (
        snapshot.get(
            "evaluation_protocol"
        )
        != protocol
    ):
        raise ValueError(
            "Snapshot evaluation "
            "protocol does not match "
            "requested protocol."
        )


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

    temporary_path.replace(
        path
    )


def load_existing_results(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
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
    protocol: str,
    per_class: int,
    seed: int,
    overwrite: bool = False,
):
    examples = load_averitec(
        DATASET_PATH
    )

    manifest = load_manifest(
        protocol,
        per_class,
        seed,
    )

    excluded_indices = set(
        manifest.get(
            "excluded_indices",
            [],
        )
    )

    eligible_examples = [
        example
        for example in examples
        if example.index
        not in excluded_indices
    ]

    selected = (
        select_stratified_examples(
            eligible_examples,
            per_class=per_class,
            seed=seed,
        )
    )

    validate_manifest(
        manifest=manifest,
        selected=selected,
        protocol=protocol,
        per_class=per_class,
        seed=seed,
    )

    output_path = (
        get_output_path(
            protocol=protocol,
            per_class=per_class,
            seed=seed,
        )
    )

    if overwrite:
        existing_results = []

    else:
        existing_results = (
            load_existing_results(
                output_path
            )
        )

    records_by_index = {}

    for record in (
        existing_results
    ):
        index = record.get(
            "index"
        )

        if index is not None:
            records_by_index[
                index
            ] = record

    print(
        f"PROTOCOL: {protocol}"
    )

    print(
        f"PER CLASS: {per_class}"
    )

    print(
        f"SEED: {seed}"
    )

    print(
        f"TOTAL: {len(selected)}"
    )

    print(
        "EXCLUDED INDICES:",
        len(
            excluded_indices
        ),
    )

    print(
        "SAMPLE ID:",
        get_sample_id(
            per_class,
            seed,
        ),
    )

    print(
        f"OUTPUT: {output_path}"
    )

    sample_root = (
        get_sample_root(
            per_class,
            seed,
        )
    )

    for number, example in enumerate(
        selected,
        start=1,
    ):
        print()

        print(
            f"[{number}/{len(selected)}]"
        )

        print(
            f"INDEX: {example.index}"
        )

        print(
            f"CLAIM: {example.claim}"
        )

        print(
            f"GOLD: {example.label}"
        )

        existing = (
            records_by_index.get(
                example.index
            )
        )

        if (
            existing is not None
            and "predicted_label"
            in existing
            and not overwrite
        ):
            print(
                "SKIP: completed result "
                "already exists."
            )

            print(
                "VERACITY: "
                f"{existing['predicted_label']}"
            )

            print(
                "OPERATIONAL: "
                f"{existing.get('operational_prediction')}"
            )

            continue

        snapshot_path = (
            get_snapshot_path(
                sample_root,
                protocol,
                example.index,
            )
        )

        if not snapshot_path.exists():
            raise FileNotFoundError(
                "Missing frozen snapshot: "
                f"{snapshot_path}. "
                "Build stratified snapshots "
                "for this sample first."
            )

        snapshot = (
            load_averitec_snapshot(
                snapshot_path
            )
        )

        validate_snapshot(
            snapshot,
            example,
            protocol,
        )

        evidence = (
            evidence_from_snapshot(
                snapshot
            )
        )

        preparation = snapshot[
            "claim_preparation"
        ]

        start = (
            time.perf_counter()
        )

        try:
            if not preparation.get(
                "is_clear"
            ):
                result = (
                    build_unclear_result(
                        preparation.get(
                            "reason"
                        )
                        or (
                            "Claim could not "
                            "be prepared "
                            "reliably."
                        )
                    )
                )

            else:
                result = (
                    analyze_claim_with_llm(
                        claim=preparation[
                            "normalized_claim"
                        ],
                        evidence=evidence,
                    )
                )

            analysis_latency_ms = round(
                (
                    time.perf_counter()
                    - start
                )
                * 1000
            )

            prediction = (
                trustnet_to_averitec(
                    result
                )
            )

            operational_prediction = (
                trustnet_operational_to_averitec(
                    result
                )
            )

            correct = (
                prediction
                == example.label
            )

            operational_correct = (
                operational_prediction
                == example.label
            )

            record = {
                "evaluation_protocol": (
                    protocol
                ),
                "sample_id": (
                    get_sample_id(
                        per_class,
                        seed,
                    )
                ),
                "index": (
                    example.index
                ),
                "claim": (
                    example.claim
                ),
                "gold_label": (
                    example.label
                ),
                "predicted_label": (
                    prediction
                ),
                "operational_prediction": (
                    operational_prediction
                ),
                "correct": (
                    correct
                ),
                "operational_correct": (
                    operational_correct
                ),
                "overall_score": (
                    result.get(
                        "overall_score"
                    )
                ),
                "credibility_label": (
                    result.get(
                        "label"
                    )
                ),
                "evidence_direction": (
                    result.get(
                        "evidence_analysis",
                        {},
                    ).get(
                        "overall_evidence_direction"
                    )
                ),
                "llm_evidence_direction": (
                    result.get(
                        "evidence_analysis",
                        {},
                    ).get(
                        "llm_overall_evidence_direction"
                    )
                ),
                "direction_was_resolved": (
                    result.get(
                        "evidence_analysis",
                        {},
                    ).get(
                        "direction_was_resolved",
                        False,
                    )
                ),
                "subclaim_assessments": (
                    result.get(
                        "evidence_analysis",
                        {},
                    ).get(
                        "subclaim_assessments",
                        [],
                    )
                ),
                "evidence_assessments": (
                    result.get(
                        "evidence_analysis",
                        {},
                    ).get(
                        "evidence_assessments",
                        [],
                    )
                ),
                "uncertainty": (
                    result.get(
                        "uncertainty"
                    )
                ),
                "factors": (
                    result.get(
                        "factors"
                    )
                ),
                "llm_timings": (
                    result.get(
                        "llm_timings"
                    )
                ),
                "analysis_latency_ms": (
                    analysis_latency_ms
                ),
                "context": (
                    snapshot.get(
                        "context"
                    )
                ),
                "retrieval_end_date": (
                    snapshot.get(
                        "retrieval_end_date"
                    )
                ),
                "search_query": (
                    preparation.get(
                        "search_query"
                    )
                ),
                "verification_queries": (
                    preparation.get(
                        "verification_queries",
                        [],
                    )
                ),
                "retrieval_config": (
                    snapshot.get(
                        "retrieval_config"
                    )
                ),
                "snapshot_path": (
                    str(
                        snapshot_path
                    )
                ),
                "snapshot_sha256": (
                    snapshot_sha256(
                        snapshot_path
                    )
                ),
                "sources": [
                    {
                        "title": (
                            item.title
                        ),
                        "url": (
                            item.url
                        ),
                        "source": (
                            item.source
                        ),
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
                f"VERACITY: {prediction}"
            )

            print(
                "OPERATIONAL: "
                f"{operational_prediction}"
            )

            print(
                "LLM DIRECTION: "
                f"{record['llm_evidence_direction']}"
            )

            print(
                "RESOLVED DIRECTION: "
                f"{record['evidence_direction']}"
            )

            print(
                "RESOLVER CHANGED: "
                f"{record['direction_was_resolved']}"
            )

            print(
                f"CORRECT: {correct}"
            )

            print(
                "OPERATIONAL CORRECT: "
                f"{operational_correct}"
            )

            print(
                "FROZEN EVIDENCE: "
                f"{len(evidence)} sources"
            )

            print(
                "ANALYSIS LATENCY: "
                f"{analysis_latency_ms} ms"
            )

        except Exception as exc:
            record = {
                "evaluation_protocol": (
                    protocol
                ),
                "sample_id": (
                    get_sample_id(
                        per_class,
                        seed,
                    )
                ),
                "index": (
                    example.index
                ),
                "claim": (
                    example.claim
                ),
                "gold_label": (
                    example.label
                ),
                "snapshot_path": (
                    str(
                        snapshot_path
                    )
                ),
                "snapshot_sha256": (
                    snapshot_sha256(
                        snapshot_path
                    )
                ),
                "error": (
                    str(
                        exc
                    )
                ),
            }

            print(
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

    successful = [
        item
        for item in results
        if "predicted_label"
        in item
    ]

    error_count = sum(
        "error" in item
        for item in results
    )

    correct_count = sum(
        item[
            "correct"
        ]
        for item in successful
    )

    operational_correct_count = sum(
        item[
            "operational_correct"
        ]
        for item in successful
    )

    resolver_changes = sum(
        bool(
            item.get(
                "direction_was_resolved"
            )
        )
        for item in successful
    )

    print()

    print(
        "===== STRATIFIED TEST SUMMARY ====="
    )

    print(
        f"Protocol: {protocol}"
    )

    print(
        "Sample:",
        get_sample_id(
            per_class,
            seed,
        ),
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "Completed: "
        f"{len(successful)}/"
        f"{len(selected)}"
    )

    print(
        f"Errors: {error_count}"
    )

    print(
        "Veracity correct: "
        f"{correct_count}/"
        f"{len(successful)}"
    )

    print(
        "Operational correct: "
        f"{operational_correct_count}/"
        f"{len(successful)}"
    )

    print(
        "Resolver changes: "
        f"{resolver_changes}/"
        f"{len(successful)}"
    )

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--protocol",
        choices=(
            EVALUATION_PROTOCOLS
        ),
        default=(
            AVERITEC_STANDARD_PROTOCOL
        ),
    )

    parser.add_argument(
        "--per-class",
        type=int,
        default=(
            DEFAULT_PER_CLASS
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_SAMPLE_SEED
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        protocol=args.protocol,
        per_class=args.per_class,
        seed=args.seed,
        overwrite=args.overwrite,
    )
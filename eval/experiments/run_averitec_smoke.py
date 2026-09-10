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
    EVALUATION_PROTOCOLS,
    TRUSTNET_HISTORICAL_PROTOCOL,
)
from eval.metrics.label_mapping import (
    trustnet_operational_to_averitec,
    trustnet_to_averitec,
)
from eval.snapshots.averitec import (
    evidence_from_snapshot,
    get_snapshot_path,
    load_averitec_snapshot,
    select_smoke_examples,
    snapshot_sha256,
)


DATASET_PATH = Path(
    "eval/datasets/averitec/dev.json"
)

SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_smoke"
)

RESULTS_DIR = Path(
    "eval/results"
)


def get_output_path(
    protocol: str,
) -> Path:
    return (
        RESULTS_DIR
        / (
            "averitec_smoke_"
            f"{protocol}.json"
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
        },
    }


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


def run(
    protocol: str,
):
    examples = load_averitec(
        DATASET_PATH
    )

    selected = (
        select_smoke_examples(
            examples
        )
    )

    output_path = (
        get_output_path(
            protocol
        )
    )

    results = []

    print(
        f"PROTOCOL: {protocol}"
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

        snapshot_path = (
            get_snapshot_path(
                SNAPSHOT_ROOT,
                protocol,
                example.index,
            )
        )

        if not snapshot_path.exists():
            raise FileNotFoundError(
                "Missing frozen snapshot: "
                f"{snapshot_path}. "
                "Build snapshots for "
                "this evaluation protocol "
                "first."
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

        results.append(
            record
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    successful = [
        item
        for item in results
        if "predicted_label" in item
    ]

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

    print()

    print(
        "===== SMOKE TEST SUMMARY ====="
    )

    print(
        f"Protocol: {protocol}"
    )

    print(
        "Completed: "
        f"{len(successful)}/"
        f"{len(results)}"
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
            TRUSTNET_HISTORICAL_PROTOCOL
        ),
    )

    args = parser.parse_args()

    run(
        protocol=args.protocol
    )
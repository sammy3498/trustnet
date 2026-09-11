import argparse
import json
from pathlib import Path

from backend.app.retrieval.search_pipeline import (
    build_search_queries,
    search_and_retrieve,
)


SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_stratified/"
    "train/n12_seed4301/averitec_standard"
)

OUTPUT_PATH = Path(
    "eval/results/"
    "retrieval_v2_2_counterfactual_"
    "1399_1998_2674.json"
)

DEFAULT_INDICES = [
    1399,
    1998,
    2674,
]

BENCHMARK_EXCLUSIONS = [
    "averitec",
]


def load_snapshot(
    index: int,
) -> dict:
    path = (
        SNAPSHOT_ROOT
        / f"{index}.json"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing snapshot: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def get_queries(
    snapshot: dict,
) -> list[str]:
    prepared = (
        snapshot[
            "claim_preparation"
        ]
    )

    return build_search_queries(
        query=prepared[
            "normalized_claim"
        ],
        alternate_query=(
            prepared.get(
                "search_query"
            )
        ),
        verification_queries=(
            prepared.get(
                "verification_queries",
                [],
            )
        ),
    )


def run(
    indices: list[int],
    dry_run: bool,
):
    snapshots = [
        load_snapshot(
            index
        )
        for index in indices
    ]

    total_queries = 0

    print(
        "===== RETRIEVAL V2.2 "
        "COUNTERFACTUAL ====="
    )

    print(
        "CASES:",
        len(
            snapshots
        ),
    )

    for snapshot in snapshots:
        index = snapshot[
            "example_index"
        ]

        queries = (
            get_queries(
                snapshot
            )
        )

        total_queries += len(
            queries
        )

        print()
        print(
            "INDEX:",
            index,
        )

        print(
            "CLEAR:",
            snapshot[
                "claim_preparation"
            ].get(
                "is_clear"
            ),
        )

        print(
            "OLD EVIDENCE:",
            len(
                snapshot.get(
                    "evidence",
                    [],
                )
            ),
        )

        print(
            "SEARCH QUERIES:",
            len(
                queries
            ),
        )

        for number, query in (
            enumerate(
                queries,
                start=1,
            )
        ):
            print(
                f"  [{number}] {query}"
            )

    print()
    print(
        "MAX TAVILY SEARCHES:",
        total_queries,
    )

    if dry_run:
        print(
            "OPENAI CALLS: 0"
        )

        print(
            "TAVILY CALLS: 0"
        )

        return

    results = []

    for number, snapshot in (
        enumerate(
            snapshots,
            start=1,
        )
    ):
        index = (
            snapshot[
                "example_index"
            ]
        )

        prepared = (
            snapshot[
                "claim_preparation"
            ]
        )

        print()
        print(
            f"[{number}/"
            f"{len(snapshots)}] "
            f"INDEX: {index}"
        )

        evidence = (
            search_and_retrieve(
                query=prepared[
                    "normalized_claim"
                ],
                alternate_query=(
                    prepared.get(
                        "search_query"
                    )
                ),
                verification_queries=(
                    prepared.get(
                        "verification_queries",
                        [],
                    )
                ),
                search_results=(
                    snapshot[
                        "retrieval_config"
                    ][
                        "search_results"
                    ]
                ),
                top_k=(
                    snapshot[
                        "retrieval_config"
                    ][
                        "top_k"
                    ]
                ),
                end_date=(
                    snapshot.get(
                        "retrieval_end_date"
                    )
                ),
                excluded_result_patterns=(
                    BENCHMARK_EXCLUSIONS
                ),
            )
        )

        print(
            "OLD EVIDENCE:",
            len(
                snapshot.get(
                    "evidence",
                    [],
                )
            ),
        )

        print(
            "NEW EVIDENCE:",
            len(
                evidence
            ),
        )

        for item_number, item in (
            enumerate(
                evidence,
                start=1,
            )
        ):
            print(
                f"  [{item_number}] "
                f"{item.evidence_origin} "
                f"{item.quality} "
                f"{item.published_date} "
                f"{item.title}"
            )

        results.append(
            {
                "index": index,
                "claim": (
                    snapshot[
                        "claim"
                    ]
                ),
                "gold_label": (
                    snapshot[
                        "gold_label"
                    ]
                ),
                "claim_preparation": (
                    prepared
                ),
                "old_evidence": (
                    snapshot.get(
                        "evidence",
                        [],
                    )
                ),
                "new_evidence": [
                    item.model_dump()
                    for item in evidence
                ],
            }
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "SAVED:",
        OUTPUT_PATH,
    )


if __name__ == "__main__":
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--indices",
        nargs="+",
        type=int,
        default=(
            DEFAULT_INDICES
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = (
        parser.parse_args()
    )

    run(
        indices=args.indices,
        dry_run=args.dry_run,
    )
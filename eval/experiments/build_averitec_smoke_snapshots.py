import argparse
from pathlib import Path

from backend.app.llm.claim_preparation import (
    prepare_claim,
)
from backend.app.retrieval.search_pipeline import (
    search_and_retrieve,
)
from eval.loaders.averitec import (
    load_averitec,
)
from eval.loaders.context import (
    EVALUATION_PROTOCOLS,
    TRUSTNET_HISTORICAL_PROTOCOL,
    build_averitec_context,
    get_averitec_retrieval_end_date,
)
from eval.snapshots.averitec import (
    get_snapshot_path,
    save_averitec_snapshot,
    select_smoke_examples,
)


DATASET_PATH = Path(
    "eval/datasets/averitec/dev.json"
)

SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_smoke"
)

SEARCH_RESULTS = 10
TOP_K = 5

AVERITEC_BENCHMARK_EXCLUSIONS = [
    "averitec",
]


def run(
    protocol: str,
    overwrite: bool = False,
):
    examples = load_averitec(
        DATASET_PATH
    )

    selected = (
        select_smoke_examples(
            examples
        )
    )

    print(
        f"PROTOCOL: {protocol}"
    )

    for number, example in enumerate(
        selected,
        start=1,
    ):
        path = get_snapshot_path(
            SNAPSHOT_ROOT,
            protocol,
            example.index,
        )

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

        if (
            path.exists()
            and not overwrite
        ):
            print(
                "SKIP: snapshot already "
                f"exists: {path}"
            )
            continue

        context = (
            build_averitec_context(
                example
            )
        )

        retrieval_end_date = (
            get_averitec_retrieval_end_date(
                example,
                protocol=protocol,
            )
        )

        prepared = prepare_claim(
            example.claim,
            context=context,
        )

        print(
            "NORMALIZED:",
            prepared.get(
                "normalized_claim"
            ),
        )

        print(
            "CLEAR:",
            prepared.get(
                "is_clear"
            ),
        )

        print(
            "SEARCH:",
            prepared.get(
                "search_query"
            ),
        )

        print(
            "VERIFY:",
            prepared.get(
                "verification_queries",
                [],
            ),
        )

        print(
            "CUTOFF:",
            retrieval_end_date,
        )

        if prepared.get(
            "is_clear"
        ):
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
                        SEARCH_RESULTS
                    ),
                    top_k=TOP_K,
                    end_date=(
                        retrieval_end_date
                    ),
                    excluded_result_patterns=(
                        AVERITEC_BENCHMARK_EXCLUSIONS
                    ),
                )
            )

        else:
            evidence = []

        save_averitec_snapshot(
            path=path,
            example=example,
            context=context,
            retrieval_end_date=(
                retrieval_end_date
            ),
            prepared_claim=prepared,
            evidence=evidence,
            search_results=(
                SEARCH_RESULTS
            ),
            top_k=TOP_K,
            evaluation_protocol=(
                protocol
            ),
        )

        print(
            "EVIDENCE COUNT: "
            f"{len(evidence)}"
        )

        print(
            f"SAVED: {path}"
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

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        protocol=args.protocol,
        overwrite=args.overwrite,
    )
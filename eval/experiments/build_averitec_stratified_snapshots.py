import argparse
import json
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
    AVERITEC_STANDARD_PROTOCOL,
    EVALUATION_PROTOCOLS,
    build_averitec_context,
    get_averitec_retrieval_end_date,
)
from eval.sampling.averitec import (
    AVERITEC_LABEL_ORDER,
    DEFAULT_SAMPLE_SEED,
    select_stratified_examples,
)
from eval.snapshots.averitec import (
    get_snapshot_path,
    save_averitec_snapshot,
)


DATASET_PATHS = {
    "dev": Path(
        "eval/datasets/averitec/dev.json"
    ),
    "train": Path(
        "eval/datasets/averitec/train.json"
    ),
}

SUPPORTED_DATASET_SPLITS = (
    "dev",
    "train",
)

SNAPSHOT_ROOT = Path(
    "eval/snapshots/averitec_stratified"
)

SEARCH_RESULTS = 10
TOP_K = 5
DEFAULT_PER_CLASS = 5

AVERITEC_BENCHMARK_EXCLUSIONS = [
    "averitec",
]


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
    dataset_split: str,
    per_class: int,
    seed: int,
) -> Path:
    sample_id = (
        get_sample_id(
            per_class,
            seed,
        )
    )

    if dataset_split == "dev":
        return (
            SNAPSHOT_ROOT
            / sample_id
        )

    return (
        SNAPSHOT_ROOT
        / dataset_split
        / sample_id
    )


def get_manifest_path(
    dataset_split: str,
    protocol: str,
    per_class: int,
    seed: int,
) -> Path:
    return (
        get_sample_root(
            dataset_split,
            per_class,
            seed,
        )
        / protocol
        / "manifest.json"
    )


def load_excluded_indices(
    manifest_path: (
        Path | None
    ),
    dataset_split: str,
) -> set[int]:
    if manifest_path is None:
        return set()

    if not (
        manifest_path.exists()
    ):
        raise FileNotFoundError(
            "Exclusion manifest not "
            f"found: {manifest_path}"
        )

    with manifest_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(
            file
        )

    manifest_split = (
        manifest.get(
            "split"
        )
    )

    if (
        manifest_split is not None
        and manifest_split
        != dataset_split
    ):
        raise ValueError(
            "Exclusion manifest split "
            "does not match requested "
            "dataset split."
        )

    examples = manifest.get(
        "examples"
    )

    if not isinstance(
        examples,
        list,
    ):
        raise ValueError(
            "Exclusion manifest does "
            "not contain an examples list."
        )

    excluded = set()

    for item in examples:
        index = item.get(
            "index"
        )

        if not isinstance(
            index,
            int,
        ):
            raise ValueError(
                "Invalid example index in "
                "exclusion manifest."
            )

        excluded.add(
            index
        )

    return excluded


def build_manifest(
    selected,
    dataset_split: str,
    protocol: str,
    per_class: int,
    seed: int,
    excluded_indices: (
        set[int]
    ),
    exclusion_manifest: (
        Path | None
    ),
) -> dict:
    return {
        "snapshot_layout_version": 2,
        "dataset": "AVeriTeC",
        "split": dataset_split,
        "evaluation_protocol": (
            protocol
        ),
        "seed": seed,
        "per_class": per_class,
        "sample_size": len(
            selected
        ),
        "sample_id": get_sample_id(
            per_class,
            seed,
        ),
        "search_results": (
            SEARCH_RESULTS
        ),
        "top_k": TOP_K,
        "benchmark_exclusions": (
            AVERITEC_BENCHMARK_EXCLUSIONS
        ),
        "excluded_indices": sorted(
            excluded_indices
        ),
        "exclusion_manifest": (
            str(
                exclusion_manifest
            )
            if exclusion_manifest
            is not None
            else None
        ),
        "examples": [
            {
                "index": (
                    example.index
                ),
                "label": (
                    example.label
                ),
                "claim": (
                    example.claim
                ),
            }
            for example
            in selected
        ],
    }


def save_manifest(
    manifest: dict,
    dataset_split: str,
    protocol: str,
    per_class: int,
    seed: int,
    overwrite: bool,
):
    path = (
        get_manifest_path(
            dataset_split,
            protocol,
            per_class,
            seed,
        )
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if (
        path.exists()
        and not overwrite
    ):
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            existing = json.load(
                file
            )

        if existing != manifest:
            raise ValueError(
                "Existing manifest differs "
                "from the requested sample. "
                "Use a different seed or "
                "explicitly use --overwrite."
            )

        print(
            f"MANIFEST EXISTS: {path}"
        )

        return

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"MANIFEST: {path}"
    )


def run(
    dataset_split: str,
    protocol: str,
    per_class: int,
    seed: int,
    overwrite: bool = False,
    exclusion_manifest: (
        Path | None
    ) = None,
):
    dataset_path = (
        DATASET_PATHS[
            dataset_split
        ]
    )

    if not (
        dataset_path.exists()
    ):
        raise FileNotFoundError(
            "Missing AVeriTeC dataset: "
            f"{dataset_path}"
        )

    examples = load_averitec(
        dataset_path,
        skip_invalid_claims=(
            dataset_split
            == "train"
        ),
    )

    excluded_indices = (
        load_excluded_indices(
            exclusion_manifest,
            dataset_split,
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

    manifest = build_manifest(
        selected=selected,
        dataset_split=(
            dataset_split
        ),
        protocol=protocol,
        per_class=per_class,
        seed=seed,
        excluded_indices=(
            excluded_indices
        ),
        exclusion_manifest=(
            exclusion_manifest
        ),
    )

    print(
        f"DATASET: {dataset_split}"
    )

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

    save_manifest(
        manifest=manifest,
        dataset_split=(
            dataset_split
        ),
        protocol=protocol,
        per_class=per_class,
        seed=seed,
        overwrite=overwrite,
    )

    sample_root = (
        get_sample_root(
            dataset_split,
            per_class,
            seed,
        )
    )

    for number, example in (
        enumerate(
            selected,
            start=1,
        )
    ):
        path = (
            get_snapshot_path(
                sample_root,
                protocol,
                example.index,
            )
        )

        print()

        print(
            f"[{number}/{len(selected)}]"
        )

        print(
            f"INDEX: {example.index}"
        )

        print(
            f"LABEL: {example.label}"
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

        prepared = (
            prepare_claim(
                example.claim,
                context=context,
            )
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
            dataset_split=(
                dataset_split
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
        "--dataset",
        choices=(
            SUPPORTED_DATASET_SPLITS
        ),
        default="dev",
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
        "--exclude-manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    run(
        dataset_split=(
            args.dataset
        ),
        protocol=args.protocol,
        per_class=args.per_class,
        seed=args.seed,
        overwrite=args.overwrite,
        exclusion_manifest=(
            args.exclude_manifest
        ),
    )
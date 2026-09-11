import argparse
import json
from collections import Counter
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
)
from eval.snapshots.averitec import (
    get_snapshot_path,
    save_averitec_snapshot,
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

CLAIM_PREPARATION_CACHE = (
    SNAPSHOT_BASE
    / "_claim_preparation_cache"
)

SEARCH_RESULTS = 10
TOP_K = 5

MAX_SEARCH_QUERIES_PER_CASE = 4

AVERITEC_BENCHMARK_EXCLUSIONS = [
    "averitec",
]

EXPECTED_SAMPLE_SIZE = 80
EXPECTED_PER_CLASS = 20
EXPECTED_SEED = 4202


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
) -> tuple[dict, list[dict]]:
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
            "Unexpected manifest size: "
            f"{len(examples)}"
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


def load_validation_manifest():
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
            "Validation manifest split "
            "must be train."
        )

    if manifest.get(
        "seed"
    ) != EXPECTED_SEED:
        raise ValueError(
            "Validation manifest seed "
            "does not match the frozen "
            "seed."
        )

    if manifest.get(
        "per_class"
    ) != EXPECTED_PER_CLASS:
        raise ValueError(
            "Validation manifest "
            "per_class does not match "
            "the frozen design."
        )

    if manifest.get(
        "sample_size"
    ) != EXPECTED_SAMPLE_SIZE:
        raise ValueError(
            "Validation manifest "
            "sample_size does not match "
            "the frozen design."
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
        record["index"]
        for record
        in validation_records
    }

    development_indices = {
        record["index"]
        for record
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


def load_dataset_by_index():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            "AVeriTeC train dataset "
            f"not found: {DATASET_PATH}"
        )

    dataset = load_averitec(
        DATASET_PATH,
        skip_invalid_claims=True,
    )

    return {
        example.index: example
        for example in dataset
    }


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
                "Validation index does "
                "not exist in the source "
                "dataset."
            )

        labels.append(
            example.label
        )

    counts = Counter(labels)

    if set(counts) != set(
        AVERITEC_LABEL_ORDER
    ):
        raise ValueError(
            "Derived validation label "
            "set does not match "
            "AVeriTeC."
        )

    if any(
        counts[label]
        != EXPECTED_PER_CLASS
        for label
        in AVERITEC_LABEL_ORDER
    ):
        raise ValueError(
            "Derived validation labels "
            "are not 20-per-class "
            "balanced."
        )


def get_claim_cache_path(
    index: int,
) -> Path:
    return (
        CLAIM_PREPARATION_CACHE
        / f"{index}.json"
    )


def load_or_prepare_claim(
    example,
):
    cache_path = (
        get_claim_cache_path(
            example.index
        )
    )

    if cache_path.exists():
        cached = load_json(
            cache_path
        )

        if (
            cached.get("index")
            != example.index
        ):
            raise ValueError(
                "Claim-preparation cache "
                "index mismatch."
            )

        context = cached.get(
            "context"
        )

        prepared = cached.get(
            "prepared_claim"
        )

        if not isinstance(
            prepared,
            dict,
        ):
            raise ValueError(
                "Invalid cached claim "
                "preparation."
            )

        return (
            context,
            prepared,
            False,
        )

    context = (
        build_averitec_context(
            example
        )
    )

    prepared = prepare_claim(
        example.claim,
        context=context,
    )

    save_json(
        cache_path,
        {
            "index": (
                example.index
            ),
            "context": context,
            "prepared_claim": (
                prepared
            ),
        },
    )

    return (
        context,
        prepared,
        True,
    )


def get_pending_records(
    records: list[dict],
    protocol: str,
) -> list[dict]:
    pending = []

    for record in records:
        path = get_snapshot_path(
            SNAPSHOT_BASE,
            protocol,
            record["index"],
        )

        if not path.exists():
            pending.append(
                record
            )

    return pending


def dry_run(
    records: list[dict],
    protocol: str,
    max_cases: int,
):
    pending = get_pending_records(
        records,
        protocol,
    )

    batch = (
        pending[:max_cases]
        if max_cases > 0
        else []
    )

    missing_claim_cache = sum(
        not get_claim_cache_path(
            record["index"]
        ).exists()
        for record in batch
    )

    ready = (
        len(records)
        - len(pending)
    )

    print(
        "===== V2 VALIDATION "
        "SNAPSHOT DRY RUN ====="
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
        "SNAPSHOTS READY:",
        ready,
    )
    print(
        "SNAPSHOTS PENDING:",
        len(pending),
    )
    print(
        "BATCH CAP:",
        max_cases,
    )
    print(
        "CASES THIS RUN:",
        len(batch),
    )
    print(
        "OPENAI CLAIM-PREP "
        "INVOCATIONS MAX:",
        missing_claim_cache,
    )
    print(
        "TAVILY SEARCHES MAX:",
        len(batch)
        * MAX_SEARCH_QUERIES_PER_CASE,
    )
    print(
        "EVIDENCE ANALYSIS "
        "CALLS:",
        0,
    )
    print(
        "CLAIMS OR INDICES "
        "PRINTED:",
        0,
    )


def run(
    records: list[dict],
    dataset_by_index: dict,
    protocol: str,
    max_cases: int,
    confirm_live: bool,
):
    if max_cases <= 0:
        raise ValueError(
            "--max-cases must be "
            "greater than zero for "
            "a live run."
        )

    if not confirm_live:
        raise ValueError(
            "Live validation snapshot "
            "building requires "
            "--confirm-live."
        )

    pending = get_pending_records(
        records,
        protocol,
    )

    batch = pending[
        :max_cases
    ]

    missing_claim_cache = sum(
        not get_claim_cache_path(
            record["index"]
        ).exists()
        for record in batch
    )

    print(
        "===== V2 VALIDATION "
        "SNAPSHOT BUILD ====="
    )
    print(
        "PROTOCOL:",
        protocol,
    )
    print(
        "TOTAL VALIDATION CASES:",
        len(records),
    )
    print(
        "PENDING BEFORE RUN:",
        len(pending),
    )
    print(
        "CASES ALLOWED THIS RUN:",
        len(batch),
    )
    print(
        "OPENAI CLAIM-PREP "
        "INVOCATIONS MAX:",
        missing_claim_cache,
    )
    print(
        "TAVILY SEARCHES MAX:",
        len(batch)
        * MAX_SEARCH_QUERIES_PER_CASE,
    )
    print(
        "CLAIMS OR INDICES "
        "PRINTED:",
        0,
    )

    if not batch:
        print(
            "NO PENDING SNAPSHOTS."
        )
        return

    claim_prep_calls = 0
    completed = 0

    for number, record in (
        enumerate(
            batch,
            start=1,
        )
    ):
        example = (
            dataset_by_index[
                record["index"]
            ]
        )

        print()
        print(
            f"[{number}/"
            f"{len(batch)}] "
            "processing"
        )

        (
            context,
            prepared,
            claim_prep_called,
        ) = load_or_prepare_claim(
            example
        )

        if claim_prep_called:
            claim_prep_calls += 1

        retrieval_end_date = (
            get_averitec_retrieval_end_date(
                example,
                protocol=protocol,
            )
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

        path = get_snapshot_path(
            SNAPSHOT_BASE,
            protocol,
            example.index,
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
            dataset_split="train",
        )

        completed += 1

        print(
            "EVIDENCE COUNT:",
            len(evidence),
        )
        print(
            "SNAPSHOT SAVED:",
            True,
        )

    remaining = len(
        get_pending_records(
            records,
            protocol,
        )
    )

    print()
    print(
        "===== RUN COMPLETE ====="
    )
    print(
        "SNAPSHOTS BUILT:",
        completed,
    )
    print(
        "OPENAI CLAIM-PREP "
        "INVOCATIONS THIS RUN:",
        claim_prep_calls,
    )
    print(
        "PENDING AFTER RUN:",
        remaining,
    )
    print(
        "EVIDENCE ANALYSIS "
        "CALLS:",
        0,
    )


def main():
    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--protocol",
        choices=EVALUATION_PROTOCOLS,
        default=(
            AVERITEC_STANDARD_PROTOCOL
        ),
    )

    parser.add_argument(
        "--max-cases",
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
        load_validation_manifest()
    )

    verify_development_disjoint(
        records
    )

    dataset_by_index = (
        load_dataset_by_index()
    )

    validate_against_dataset(
        records,
        dataset_by_index,
    )

    if args.dry_run:
        dry_run(
            records=records,
            protocol=args.protocol,
            max_cases=args.max_cases,
        )
        return

    run(
        records=records,
        dataset_by_index=(
            dataset_by_index
        ),
        protocol=args.protocol,
        max_cases=args.max_cases,
        confirm_live=(
            args.confirm_live
        ),
    )


if __name__ == "__main__":
    main()
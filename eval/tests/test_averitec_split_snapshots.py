from types import SimpleNamespace

import pytest

from eval.snapshots.averitec import (
    load_averitec_snapshot,
    save_averitec_snapshot,
)


def build_example():
    return SimpleNamespace(
        index=7,
        claim="Example claim",
        label="Supported",
    )


def build_prepared_claim():
    return {
        "normalized_claim": (
            "Example claim"
        ),
        "is_clear": True,
        "reason": (
            "The claim is clear."
        ),
        "search_query": (
            "example claim"
        ),
        "verification_queries": [
            "example claim evidence"
        ],
    }


def test_train_snapshot_records_train_split(
    tmp_path,
):
    path = (
        tmp_path
        / "snapshot.json"
    )

    save_averitec_snapshot(
        path=path,
        example=build_example(),
        context="",
        retrieval_end_date=None,
        prepared_claim=(
            build_prepared_claim()
        ),
        evidence=[],
        search_results=10,
        top_k=5,
        evaluation_protocol=(
            "averitec_standard"
        ),
        dataset_split="train",
    )

    snapshot = (
        load_averitec_snapshot(
            path
        )
    )

    assert (
        snapshot[
            "split"
        ]
        == "train"
    )


def test_invalid_snapshot_split_rejected(
    tmp_path,
):
    path = (
        tmp_path
        / "snapshot.json"
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported AVeriTeC "
            "dataset split"
        ),
    ):
        save_averitec_snapshot(
            path=path,
            example=build_example(),
            context="",
            retrieval_end_date=None,
            prepared_claim=(
                build_prepared_claim()
            ),
            evidence=[],
            search_results=10,
            top_k=5,
            evaluation_protocol=(
                "averitec_standard"
            ),
            dataset_split="invalid",
        )
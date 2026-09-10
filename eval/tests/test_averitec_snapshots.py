import json
from pathlib import Path

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from eval.loaders.averitec import (
    AVeriTeCExample,
)
from eval.loaders.context import (
    AVERITEC_STANDARD_PROTOCOL,
    TRUSTNET_HISTORICAL_PROTOCOL,
)
from eval.snapshots.averitec import (
    evidence_from_snapshot,
    get_snapshot_path,
    load_averitec_snapshot,
    save_averitec_snapshot,
    snapshot_sha256,
)


def make_example():
    return AVeriTeCExample(
        index=7,
        claim="Example claim",
        label="Supported",
        raw={},
    )


def test_snapshot_round_trip(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "snapshot.json"
    )

    evidence = [
        RetrievedEvidence(
            title="Example",
            url=(
                "https://example.com/a"
            ),
            source="example.com",
            text="Evidence text",
            quality="medium",
            published_date=(
                "2020-01-01"
            ),
        )
    ]

    save_averitec_snapshot(
        path=path,
        example=make_example(),
        context="Speaker: Example",
        retrieval_end_date=(
            "2020-01-02"
        ),
        prepared_claim={
            "normalized_claim": (
                "Example claim"
            ),
            "is_clear": True,
            "reason": "Clear claim.",
            "search_query": (
                "example search"
            ),
            "verification_queries": [
                "example verify"
            ],
        },
        evidence=evidence,
        search_results=10,
        top_k=5,
        evaluation_protocol=(
            TRUSTNET_HISTORICAL_PROTOCOL
        ),
    )

    snapshot = (
        load_averitec_snapshot(
            path
        )
    )

    assert (
        snapshot[
            "evaluation_protocol"
        ]
        == TRUSTNET_HISTORICAL_PROTOCOL
    )

    assert (
        snapshot[
            "example_index"
        ]
        == 7
    )


def test_snapshot_restores_evidence(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "snapshot.json"
    )

    save_averitec_snapshot(
        path=path,
        example=make_example(),
        context="",
        retrieval_end_date=None,
        prepared_claim={
            "normalized_claim": (
                "Example claim"
            ),
            "is_clear": True,
            "reason": "",
            "search_query": "example",
            "verification_queries": [],
        },
        evidence=[
            RetrievedEvidence(
                title="Source",
                url=(
                    "https://example.com"
                ),
                source="example.com",
                text="Evidence",
                quality="low",
            )
        ],
        search_results=10,
        top_k=5,
        evaluation_protocol=(
            AVERITEC_STANDARD_PROTOCOL
        ),
    )

    snapshot = (
        load_averitec_snapshot(
            path
        )
    )

    evidence = (
        evidence_from_snapshot(
            snapshot
        )
    )

    assert len(evidence) == 1

    assert isinstance(
        evidence[0],
        RetrievedEvidence,
    )


def test_protocols_use_separate_paths(
    tmp_path: Path,
):
    historical = (
        get_snapshot_path(
            tmp_path,
            TRUSTNET_HISTORICAL_PROTOCOL,
            7,
        )
    )

    standard = (
        get_snapshot_path(
            tmp_path,
            AVERITEC_STANDARD_PROTOCOL,
            7,
        )
    )

    assert historical != standard


def test_legacy_snapshot_is_historical(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "legacy.json"
    )

    data = {
        "snapshot_version": 1,
        "dataset": "AVeriTeC",
        "split": "dev",
        "example_index": 7,
        "claim": "Example claim",
        "claim_preparation": {},
        "retrieval_config": {},
        "evidence": [],
    }

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    snapshot = (
        load_averitec_snapshot(
            path
        )
    )

    assert (
        snapshot[
            "evaluation_protocol"
        ]
        == TRUSTNET_HISTORICAL_PROTOCOL
    )


def test_snapshot_hash_is_stable(
    tmp_path: Path,
):
    path = (
        tmp_path
        / "file.json"
    )

    path.write_text(
        '{"test": true}',
        encoding="utf-8",
    )

    first = snapshot_sha256(
        path
    )

    second = snapshot_sha256(
        path
    )

    assert first == second
    assert len(first) == 64
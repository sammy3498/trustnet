import hashlib
import json
from pathlib import Path

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from eval.loaders.context import (
    TRUSTNET_HISTORICAL_PROTOCOL,
)


SNAPSHOT_VERSION = 2

SUPPORTED_SNAPSHOT_VERSIONS = {
    1,
    2,
}

LABEL_ORDER = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]


def select_smoke_examples(
    examples,
):
    selected = []

    for label in LABEL_ORDER:
        example = next(
            item
            for item in examples
            if item.label == label
        )

        selected.append(
            example
        )

    return selected


def get_snapshot_path(
    snapshot_root: Path,
    protocol: str,
    example_index: int,
) -> Path:
    return (
        snapshot_root
        / protocol
        / f"{example_index}.json"
    )


def save_averitec_snapshot(
    path: Path,
    example,
    context: str,
    retrieval_end_date: str | None,
    prepared_claim: dict,
    evidence: list[
        RetrievedEvidence
    ],
    search_results: int,
    top_k: int,
    evaluation_protocol: str,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = {
        "snapshot_version": (
            SNAPSHOT_VERSION
        ),
        "evaluation_protocol": (
            evaluation_protocol
        ),
        "dataset": "AVeriTeC",
        "split": "dev",
        "example_index": (
            example.index
        ),
        "claim": (
            example.claim
        ),
        "gold_label": (
            example.label
        ),
        "context": context,
        "retrieval_end_date": (
            retrieval_end_date
        ),
        "claim_preparation": {
            "normalized_claim": (
                prepared_claim.get(
                    "normalized_claim"
                )
            ),
            "is_clear": (
                prepared_claim.get(
                    "is_clear"
                )
            ),
            "reason": (
                prepared_claim.get(
                    "reason"
                )
            ),
            "search_query": (
                prepared_claim.get(
                    "search_query"
                )
            ),
            "verification_queries": (
                prepared_claim.get(
                    "verification_queries",
                    [],
                )
            ),
        },
        "retrieval_config": {
            "search_results": (
                search_results
            ),
            "top_k": top_k,
        },
        "evidence": [
            item.model_dump()
            for item in evidence
        ],
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )


def load_averitec_snapshot(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    version = data.get(
        "snapshot_version"
    )

    if (
        version
        not in
        SUPPORTED_SNAPSHOT_VERSIONS
    ):
        raise ValueError(
            "Unsupported snapshot version."
        )

    if version == 1:
        data[
            "evaluation_protocol"
        ] = (
            TRUSTNET_HISTORICAL_PROTOCOL
        )

    required = [
        "evaluation_protocol",
        "dataset",
        "split",
        "example_index",
        "claim",
        "claim_preparation",
        "retrieval_config",
        "evidence",
    ]

    for key in required:
        if key not in data:
            raise ValueError(
                "Snapshot missing required field: "
                f"{key}"
            )

    return data


def evidence_from_snapshot(
    snapshot: dict,
) -> list[RetrievedEvidence]:
    return [
        RetrievedEvidence.model_validate(
            item
        )
        for item in snapshot[
            "evidence"
        ]
    ]


def snapshot_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:
        while True:
            chunk = file.read(
                65536
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()
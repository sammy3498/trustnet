import json

import pytest

from eval.loaders.averitec import (
    load_averitec,
)


def test_loader_preserves_original_indices(
    tmp_path,
):
    path = (
        tmp_path
        / "averitec.json"
    )

    data = [
        {
            "claim": "First claim",
            "label": "Supported",
        },
        {
            "claim": "",
            "label": "Refuted",
        },
        {
            "claim": "Third claim",
            "label": "Refuted",
        },
    ]

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    examples = load_averitec(
        path,
        skip_invalid_claims=True,
    )

    assert [
        item.index
        for item in examples
    ] == [
        0,
        2,
    ]


def test_loader_is_strict_by_default(
    tmp_path,
):
    path = (
        tmp_path
        / "averitec.json"
    )

    data = [
        {
            "claim": "",
            "label": "Supported",
        },
    ]

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Missing or invalid claim",
    ):
        load_averitec(
            path
        )


def test_loader_can_skip_invalid_claim(
    tmp_path,
):
    path = (
        tmp_path
        / "averitec.json"
    )

    data = [
        {
            "claim": "",
            "label": "Supported",
        },
        {
            "claim": "Valid claim",
            "label": "Supported",
        },
    ]

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )

    examples = load_averitec(
        path,
        skip_invalid_claims=True,
    )

    assert len(
        examples
    ) == 1

    assert (
        examples[0].index
        == 1
    )

    assert (
        examples[0].claim
        == "Valid claim"
    )
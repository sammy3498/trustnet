import pytest

from eval.loaders.averitec import (
    AVeriTeCExample,
)
from eval.sampling.averitec import (
    AVERITEC_LABEL_ORDER,
    select_stratified_examples,
)


def make_examples(
    count_per_class: int = 8,
):
    examples = []

    index = 0

    for label in (
        AVERITEC_LABEL_ORDER
    ):
        for _ in range(
            count_per_class
        ):
            examples.append(
                AVeriTeCExample(
                    index=index,
                    claim=(
                        f"Claim {index}"
                    ),
                    label=label,
                    raw={},
                )
            )

            index += 1

    return examples


def test_stratified_sample_size():
    result = (
        select_stratified_examples(
            make_examples(),
            per_class=5,
            seed=42,
        )
    )

    assert len(result) == 20


def test_stratified_sample_has_equal_classes():
    result = (
        select_stratified_examples(
            make_examples(),
            per_class=5,
            seed=42,
        )
    )

    counts = {
        label: 0
        for label in (
            AVERITEC_LABEL_ORDER
        )
    }

    for example in result:
        counts[
            example.label
        ] += 1

    assert counts == {
        label: 5
        for label in (
            AVERITEC_LABEL_ORDER
        )
    }


def test_stratified_sample_is_deterministic():
    examples = make_examples()

    first = (
        select_stratified_examples(
            examples,
            per_class=5,
            seed=42,
        )
    )

    second = (
        select_stratified_examples(
            examples,
            per_class=5,
            seed=42,
        )
    )

    assert [
        item.index
        for item in first
    ] == [
        item.index
        for item in second
    ]


def test_stratified_sample_is_interleaved():
    result = (
        select_stratified_examples(
            make_examples(),
            per_class=2,
            seed=42,
        )
    )

    assert [
        item.label
        for item in result[:4]
    ] == list(
        AVERITEC_LABEL_ORDER
    )

    assert [
        item.label
        for item in result[4:8]
    ] == list(
        AVERITEC_LABEL_ORDER
    )


def test_stratified_sample_rejects_insufficient_class():
    with pytest.raises(
        ValueError
    ):
        select_stratified_examples(
            make_examples(
                count_per_class=2
            ),
            per_class=5,
            seed=42,
        )


def test_stratified_sample_rejects_zero():
    with pytest.raises(
        ValueError
    ):
        select_stratified_examples(
            make_examples(),
            per_class=0,
            seed=42,
        )
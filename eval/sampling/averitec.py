import random

from eval.loaders.averitec import (
    AVeriTeCExample,
)


AVERITEC_LABEL_ORDER = (
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
)

DEFAULT_SAMPLE_SEED = 42


def select_stratified_examples(
    examples: list[AVeriTeCExample],
    per_class: int = 5,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> list[AVeriTeCExample]:
    if per_class <= 0:
        raise ValueError(
            "per_class must be greater than zero."
        )

    rng = random.Random(
        seed
    )

    sampled_by_label = {}

    for label in AVERITEC_LABEL_ORDER:
        candidates = [
            example
            for example in examples
            if example.label == label
        ]

        if len(candidates) < per_class:
            raise ValueError(
                "Not enough examples for label "
                f"{label}: requested {per_class}, "
                f"found {len(candidates)}."
            )

        sampled = rng.sample(
            candidates,
            per_class,
        )

        sampled.sort(
            key=lambda example: (
                example.index
            )
        )

        sampled_by_label[
            label
        ] = sampled

    selected = []

    for position in range(
        per_class
    ):
        for label in (
            AVERITEC_LABEL_ORDER
        ):
            selected.append(
                sampled_by_label[
                    label
                ][position]
            )

    return selected
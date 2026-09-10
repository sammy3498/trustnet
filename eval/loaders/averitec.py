import json
from dataclasses import dataclass
from pathlib import Path


VALID_LABELS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}


@dataclass
class AVeriTeCExample:
    index: int
    claim: str
    label: str
    raw: dict


def load_averitec(
    path: Path | str,
) -> list[AVeriTeCExample]:
    path = Path(
        path
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(
            file
        )

    if not isinstance(
        data,
        list,
    ):
        raise ValueError(
            "AVeriTeC dataset must be a JSON list."
        )

    examples = []

    for index, item in enumerate(
        data
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                f"Invalid AVeriTeC item at index {index}."
            )

        claim = item.get(
            "claim"
        )

        label = item.get(
            "label"
        )

        if (
            not isinstance(
                claim,
                str,
            )
            or not claim.strip()
        ):
            raise ValueError(
                f"Missing or invalid claim at index {index}."
            )

        if label not in VALID_LABELS:
            raise ValueError(
                "Invalid AVeriTeC label "
                f"at index {index}: {label}"
            )

        examples.append(
            AVeriTeCExample(
                index=index,
                claim=claim,
                label=label,
                raw=item,
            )
        )

    return examples
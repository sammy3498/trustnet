from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval.loaders.averitec import (
        AVeriTeCExample,
    )


AVERITEC_STANDARD_PROTOCOL = (
    "averitec_standard"
)

TRUSTNET_HISTORICAL_PROTOCOL = (
    "trustnet_historical_cutoff"
)

EVALUATION_PROTOCOLS = (
    AVERITEC_STANDARD_PROTOCOL,
    TRUSTNET_HISTORICAL_PROTOCOL,
)


def build_averitec_context(
    example: "AVeriTeCExample",
) -> str:
    parts = []

    speaker = example.raw.get(
        "speaker"
    )

    claim_date = example.raw.get(
        "claim_date"
    )

    location = example.raw.get(
        "location_ISO_code"
    )

    reporting_source = (
        example.raw.get(
            "reporting_source"
        )
    )

    if speaker:
        parts.append(
            f"Speaker: {speaker}"
        )

    if claim_date:
        parts.append(
            f"Claim date: {claim_date}"
        )

    if location:
        parts.append(
            f"Location: {location}"
        )

    if reporting_source:
        parts.append(
            f"Reporting source: "
            f"{reporting_source}"
        )

    return "\n".join(
        parts
    )


def build_contextualized_claim(
    example: "AVeriTeCExample",
) -> str:
    context = (
        build_averitec_context(
            example
        )
    )

    if not context:
        return example.claim

    return (
        "Context for interpreting "
        "the claim:\n"
        f"{context}\n\n"
        "Claim to fact-check:\n"
        f"{example.claim}"
    )


def get_claim_date_iso(
    example: "AVeriTeCExample",
) -> str | None:
    claim_date = example.raw.get(
        "claim_date"
    )

    if not claim_date:
        return None

    try:
        parsed = datetime.strptime(
            claim_date,
            "%d-%m-%Y",
        )

    except ValueError as exc:
        raise ValueError(
            "Invalid AVeriTeC claim date: "
            f"{claim_date}"
        ) from exc

    return (
        parsed.date().isoformat()
    )


def get_averitec_retrieval_end_date(
    example: "AVeriTeCExample",
    protocol: str = (
        TRUSTNET_HISTORICAL_PROTOCOL
    ),
) -> str | None:
    if (
        protocol
        == AVERITEC_STANDARD_PROTOCOL
    ):
        return None

    if (
        protocol
        == TRUSTNET_HISTORICAL_PROTOCOL
    ):
        return get_claim_date_iso(
            example
        )

    raise ValueError(
        "Unknown evaluation protocol: "
        f"{protocol}"
    )
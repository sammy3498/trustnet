AVERITEC_LABELS = {
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
}


DIRECTION_TO_AVERITEC = {
    "supports": "Supported",
    "contradicts": "Refuted",
    "mixed": "Conflicting Evidence/Cherrypicking",
    "insufficient": "Not Enough Evidence",
}


def trustnet_to_averitec(result: dict) -> str:
    evidence_analysis = result.get(
        "evidence_analysis",
        {},
    )

    direction = evidence_analysis.get(
        "overall_evidence_direction"
    )

    if direction is None:
        return "Not Enough Evidence"

    if direction not in DIRECTION_TO_AVERITEC:
        raise ValueError(
            f"Unknown TrustNet evidence direction: {direction}"
        )

    return DIRECTION_TO_AVERITEC[direction]


def trustnet_operational_to_averitec(
    result: dict,
) -> str:
    uncertainty = result.get(
        "uncertainty",
        {},
    )

    if uncertainty.get("gate_triggered", False):
        return "Not Enough Evidence"

    # A grounded mixed verdict is a first-class TrustNet result; do not
    # reclassify it as an abstention simply because no numeric score exists.
    return trustnet_to_averitec(result)
def detect_conflicting_evidence(
    evidence_assessments: list[dict],
    reliable_urls: set[str] | None = None,
) -> bool:
    stances = set()

    for item in evidence_assessments:
        stance = item.get(
            "stance"
        )

        if stance not in {
            "supports",
            "contradicts",
        }:
            continue

        if reliable_urls is not None:
            url = item.get(
                "url"
            )

            if url not in reliable_urls:
                continue

        stances.add(
            stance
        )

    return (
        "supports" in stances
        and "contradicts" in stances
    )
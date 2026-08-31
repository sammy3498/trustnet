def detect_conflicting_evidence(
    evidence_assessments: list[dict],
) -> bool:
    stances = {
        item["stance"]
        for item in evidence_assessments
        if item["stance"] in {"supports", "contradicts"}
    }

    return (
        "supports" in stances
        and "contradicts" in stances
    )
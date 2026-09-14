def normalize_claim_text(
    claim: str,
) -> str:
    return " ".join(
        claim.split()
    ).strip()


def prepare_claim(
    claim: str,
    context: str | None = None,
) -> dict:
    normalized_claim = (
        normalize_claim_text(
            claim
        )
    )

    is_clear = bool(
        normalized_claim
    )

    if normalized_claim:
        search_query = (
            f"{normalized_claim} "
            "official primary source"
        )

    else:
        search_query = ""

    if is_clear:
        reason = (
            "Normal Mode preserved the "
            "original factual wording and "
            "prioritized authoritative "
            "retrieval."
        )

    else:
        reason = (
            "The claim is empty."
        )

    return {
        "normalized_claim": (
            normalized_claim
        ),
        "is_clear": is_clear,
        "reason": reason,
        "search_query": (
            search_query
        ),
        "verification_queries": [],
        "timed_out": False,
    }
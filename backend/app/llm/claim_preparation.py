import re


MAX_DECOMPOSED_SUBCLAIMS = 2
MAX_SUBCLAIM_WORDS = 14


UNSAFE_COMPOUND_MARKERS = (
    " or ",
    " because ",
    " while ",
    " whereas ",
    " although ",
    " but ",
    " unless ",
    " if ",
    " therefore ",
    " however ",
    " compared ",
    " than ",
)


DEPENDENT_SUBJECT_WORDS = {
    "it",
    "they",
    "he",
    "she",
    "this",
    "that",
    "these",
    "those",
    "which",
    "who",
    "whose",
}


EXPLICIT_VERBS = {
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "has",
    "have",
    "had",
    "does",
    "do",
    "did",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
}


def normalize_claim_text(
    claim: str,
) -> str:
    return " ".join(
        claim.split()
    ).strip()


def strip_terminal_punctuation(
    text: str,
) -> str:
    return text.strip().rstrip(
        ".!?"
    ).strip()


def ensure_terminal_period(
    text: str,
) -> str:
    value = (
        strip_terminal_punctuation(
            text
        )
    )

    if not value:
        return ""

    return (
        value
        + "."
    )


def get_words(
    text: str,
) -> list[str]:
    return re.findall(
        r"[A-Za-z][A-Za-z'-]*",
        text,
    )


def has_explicit_predicate(
    clause: str,
) -> bool:
    words = get_words(
        clause
    )

    if len(
        words
    ) < 3:
        return False

    for index in range(
        1,
        len(
            words
        ) - 1,
    ):
        original = words[
            index
        ]

        lower = (
            original.lower()
        )

        if lower in EXPLICIT_VERBS:
            return True

        if (
            original == lower
            and (
                lower.endswith(
                    "ed"
                )
                or (
                    len(lower) > 3
                    and lower.endswith(
                        "s"
                    )
                )
            )
        ):
            return True

    return False


def is_safe_atomic_subclaim(
    clause: str,
) -> bool:
    value = (
        strip_terminal_punctuation(
            clause
        )
    )

    if not value:
        return False

    words = get_words(
        value
    )

    if (
        len(
            words
        ) < 3
    ):
        return False

    if (
        len(
            words
        )
        > MAX_SUBCLAIM_WORDS
    ):
        return False

    if any(
        symbol in value
        for symbol in (
            ";",
            ":",
            "%",
            "$",
            "€",
            "₹",
            "\n",
        )
    ):
        return False

    padded = (
        " "
        + value.lower()
        + " "
    )

    for marker in (
        UNSAFE_COMPOUND_MARKERS
    ):
        if marker in padded:
            return False

    first_word = (
        words[
            0
        ].lower()
    )

    if (
        first_word
        in DEPENDENT_SUBJECT_WORDS
    ):
        return False

    return has_explicit_predicate(
        value
    )


def decompose_safe_conjunction(
    claim: str,
) -> list[str]:
    normalized = (
        normalize_claim_text(
            claim
        )
    )

    if not normalized:
        return []

    if "," in normalized:
        return []

    padded = (
        " "
        + normalized.lower()
        + " "
    )

    for marker in (
        UNSAFE_COMPOUND_MARKERS
    ):
        if marker in padded:
            return []

    parts = re.split(
        r"\s+and\s+",
        normalized,
        flags=re.IGNORECASE,
    )

    if (
        len(
            parts
        )
        != MAX_DECOMPOSED_SUBCLAIMS
    ):
        return []

    subclaims = [
        ensure_terminal_period(
            part
        )
        for part in parts
    ]

    if not all(
        is_safe_atomic_subclaim(
            subclaim
        )
        for subclaim
        in subclaims
    ):
        return []

    return subclaims


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

    verification_queries = []

    if is_clear:
        verification_queries = (
            decompose_safe_conjunction(
                normalized_claim
            )
        )

        search_query = (
            f"{normalized_claim} "
            "official primary source"
        )

        if verification_queries:
            reason = (
                "Normal Mode preserved the "
                "original factual wording and "
                "identified independent atomic "
                "subclaims for verification."
            )

        else:
            reason = (
                "Normal Mode preserved the "
                "original factual wording and "
                "prioritized authoritative "
                "retrieval."
            )

    else:
        search_query = ""

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
        "verification_queries": (
            verification_queries
        ),
        "timed_out": False,
    }
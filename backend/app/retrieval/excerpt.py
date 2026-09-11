import re

from rank_bm25 import BM25Okapi


DEFAULT_PASSAGE_MAX_CHARS = 1500

MIN_PARAGRAPH_CHARS = 40


def tokenize(
    text: str,
) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def normalize_segment(
    text: str,
) -> str:
    return re.sub(
        r"[ \t]+",
        " ",
        text,
    ).strip()


def split_long_segment(
    segment: str,
    max_chars: int,
) -> list[str]:
    segment = normalize_segment(
        segment
    )

    if not segment:
        return []

    if len(segment) <= max_chars:
        return [
            segment
        ]

    sentences = [
        normalize_segment(
            sentence
        )
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            segment,
        )
        if normalize_segment(
            sentence
        )
    ]

    if (
        len(sentences) <= 1
    ):
        return [
            segment[
                start:
                start + max_chars
            ]
            for start in range(
                0,
                len(segment),
                max_chars,
            )
            if segment[
                start:
                start + max_chars
            ].strip()
        ]

    chunks = []
    current = []

    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(
                    " ".join(
                        current
                    )
                )
                current = []

            chunks.extend(
                [
                    sentence[
                        start:
                        start + max_chars
                    ]
                    for start in range(
                        0,
                        len(sentence),
                        max_chars,
                    )
                    if sentence[
                        start:
                        start + max_chars
                    ].strip()
                ]
            )

            continue

        candidate = (
            " ".join(
                current
                + [
                    sentence
                ]
            )
        )

        if (
            current
            and len(candidate)
            > max_chars
        ):
            chunks.append(
                " ".join(
                    current
                )
            )

            current = [
                sentence
            ]

        else:
            current.append(
                sentence
            )

    if current:
        chunks.append(
            " ".join(
                current
            )
        )

    return chunks


def extract_units(
    text: str,
    max_chars: int,
) -> list[str]:
    normalized = (
        text.replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )

    raw_segments = [
        normalize_segment(
            segment
        )
        for segment in re.split(
            r"\n+",
            normalized,
        )
        if normalize_segment(
            segment
        )
    ]

    useful_segments = [
        segment
        for segment
        in raw_segments
        if len(segment)
        >= MIN_PARAGRAPH_CHARS
    ]

    if not useful_segments:
        useful_segments = (
            raw_segments
        )

    units = []

    for segment in useful_segments:
        units.extend(
            split_long_segment(
                segment=segment,
                max_chars=max_chars,
            )
        )

    return [
        unit
        for unit in units
        if unit.strip()
    ]


def build_passage_windows(
    text: str,
    max_chars: int = (
        DEFAULT_PASSAGE_MAX_CHARS
    ),
) -> list[str]:
    if max_chars <= 0:
        raise ValueError(
            "max_chars must be "
            "greater than 0."
        )

    stripped = text.strip()

    if not stripped:
        return []

    if len(stripped) <= max_chars:
        return [
            stripped
        ]

    units = extract_units(
        text=stripped,
        max_chars=max_chars,
    )

    if not units:
        return [
            stripped[
                :max_chars
            ]
        ]

    passages = []
    seen = set()

    for start in range(
        len(units)
    ):
        selected = []
        current_length = 0

        for unit in units[
            start:
        ]:
            separator_length = (
                2
                if selected
                else 0
            )

            new_length = (
                current_length
                + separator_length
                + len(unit)
            )

            if (
                selected
                and new_length
                > max_chars
            ):
                break

            if (
                not selected
                and len(unit)
                > max_chars
            ):
                unit = unit[
                    :max_chars
                ]

            selected.append(
                unit
            )

            current_length = (
                sum(
                    len(item)
                    for item
                    in selected
                )
                + 2
                * (
                    len(selected)
                    - 1
                )
            )

            if (
                current_length
                >= max_chars
            ):
                break

        if not selected:
            continue

        passage = (
            "\n\n".join(
                selected
            )
        ).strip()

        if not passage:
            continue

        if len(passage) > max_chars:
            passage = passage[
                :max_chars
            ].rstrip()

        if passage in seen:
            continue

        seen.add(
            passage
        )

        passages.append(
            passage
        )

    if not passages:
        return [
            stripped[
                :max_chars
            ]
        ]

    return passages


def passage_query_coverage(
    passage: str,
    query: str,
) -> float:
    query_tokens = set(
        tokenize(
            query
        )
    )

    if not query_tokens:
        return 0.0

    passage_tokens = set(
        tokenize(
            passage
        )
    )

    overlap = (
        query_tokens
        & passage_tokens
    )

    return (
        len(overlap)
        / len(query_tokens)
    )


def get_relevant_excerpt(
    text: str,
    query: str,
    max_chars: int = (
        DEFAULT_PASSAGE_MAX_CHARS
    ),
) -> str:
    passages = (
        build_passage_windows(
            text=text,
            max_chars=max_chars,
        )
    )

    if not passages:
        return ""

    if len(passages) == 1:
        return passages[0]

    query_tokens = tokenize(
        query
    )

    if not query_tokens:
        return passages[0]

    tokenized_passages = [
        tokenize(
            passage
        )
        for passage in passages
    ]

    if not any(
        tokenized_passages
    ):
        return passages[0]

    bm25 = BM25Okapi(
        tokenized_passages
    )

    scores = bm25.get_scores(
        query_tokens
    )

    ranked = sorted(
        [
            (
                passage,
                float(score),
                passage_query_coverage(
                    passage=passage,
                    query=query,
                ),
                index,
            )
            for index, (
                passage,
                score,
            )
            in enumerate(
                zip(
                    passages,
                    scores,
                )
            )
        ],
        key=lambda item: (
            item[1]
            + item[2],
            item[2],
            -item[3],
        ),
        reverse=True,
    )

    return ranked[0][0]
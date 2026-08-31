import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def get_relevant_excerpt(
    text: str,
    query: str,
    max_chars: int = 3000,
) -> str:
    if len(text) <= max_chars:
        return text

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n")
        if len(paragraph.strip()) >= 40
    ]

    if not paragraphs:
        return text[:max_chars]

    tokenized = [
        tokenize(paragraph)
        for paragraph in paragraphs
    ]

    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(paragraphs, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    selected = []
    length = 0

    for paragraph, _ in ranked:
        if length + len(paragraph) > max_chars:
            continue

        selected.append(paragraph)
        length += len(paragraph)

        if length >= max_chars * 0.8:
            break

    if not selected:
        return text[:max_chars]

    return "\n\n".join(selected)
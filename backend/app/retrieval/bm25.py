import re

from rank_bm25 import BM25Okapi

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


def tokenize(
    text: str,
) -> list[str]:
    return re.findall(
        r"\b\w+\b",
        text.lower(),
    )


def quality_bonus(
    quality: str,
) -> float:
    bonuses = {
        "high": 2.0,
        "medium": 1.0,
        "low": 0.0,
    }

    return bonuses[
        quality
    ]


def get_document_passage(
    document: RetrievedEvidence,
    query: str,
    passages_by_url: (
        dict[str, str] | None
    ) = None,
) -> str:
    if passages_by_url is not None:
        cached_passage = (
            passages_by_url.get(
                document.url
            )
        )

        if cached_passage is not None:
            return cached_passage

    return get_relevant_excerpt(
        text=document.text,
        query=query,
        max_chars=(
            DEFAULT_PASSAGE_MAX_CHARS
        ),
    )


def retrieve_bm25(
    query: str,
    documents: list[
        RetrievedEvidence
    ],
    top_k: int = 5,
    passages_by_url: (
        dict[str, str] | None
    ) = None,
) -> list[
    RetrievedEvidence
]:
    if not documents:
        return []

    passage_texts = [
        get_document_passage(
            document=document,
            query=query,
            passages_by_url=(
                passages_by_url
            ),
        )
        for document in documents
    ]

    tokenized_documents = [
        tokenize(
            passage
        )
        for passage in passage_texts
    ]

    bm25 = BM25Okapi(
        tokenized_documents
    )

    scores = bm25.get_scores(
        tokenize(
            query
        )
    )

    ranked = sorted(
        zip(
            documents,
            scores,
        ),
        key=lambda item: (
            item[1]
            + quality_bonus(
                item[0].quality
            )
        ),
        reverse=True,
    )

    results = []

    for (
        document,
        bm25_score,
    ) in ranked[
        :top_k
    ]:
        document.bm25_score = (
            float(
                bm25_score
            )
        )

        document.quality_bonus_score = (
            quality_bonus(
                document.quality
            )
        )

        document.retrieval_score = (
            document.bm25_score
            + document
            .quality_bonus_score
        )

        results.append(
            document
        )

    return results
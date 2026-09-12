from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.bm25 import (
    retrieve_bm25,
)
from backend.app.retrieval.dense import (
    retrieve_dense,
)
from backend.app.retrieval.temporal import (
    get_soft_temporal_fit_score,
)


BM25_WEIGHT = 0.40
SEMANTIC_WEIGHT = 0.40
QUALITY_WEIGHT = 0.10
TEMPORAL_WEIGHT = 0.10


def normalize_scores(
    values: list[float],
) -> list[float]:
    if not values:
        return []

    minimum = min(
        values
    )

    maximum = max(
        values
    )

    if maximum == minimum:
        return [
            1.0
            for _ in values
        ]

    return [
        (
            value - minimum
        )
        / (
            maximum - minimum
        )
        for value in values
    ]


def quality_score(
    quality: str,
) -> float:
    scores = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.0,
    }

    return scores[
        quality
    ]


def retrieve_hybrid(
    query: str,
    documents: list[
        RetrievedEvidence
    ],
    top_k: int = 5,
    temporal_reference_text: (
        str | None
    ) = None,
    passages_by_url: (
        dict[str, str] | None
    ) = None,
) -> list[
    RetrievedEvidence
]:
    if not documents:
        return []

    retrieve_bm25(
        query=query,
        documents=documents,
        top_k=len(
            documents
        ),
        passages_by_url=(
            passages_by_url
        ),
    )

    retrieve_dense(
        query=query,
        documents=documents,
        top_k=len(
            documents
        ),
        passages_by_url=(
            passages_by_url
        ),
    )

    bm25_values = [
        document.bm25_score
        for document in documents
    ]

    normalized_bm25 = (
        normalize_scores(
            bm25_values
        )
    )

    for (
        document,
        bm25_normalized,
    ) in zip(
        documents,
        normalized_bm25,
    ):
        semantic = max(
            0.0,
            min(
                1.0,
                document.semantic_score,
            ),
        )

        temporal = (
            get_soft_temporal_fit_score(
                reference_text=(
                    temporal_reference_text
                ),
                title=document.title,
                text=document.text,
                published_date=(
                    document.published_date
                ),
            )
        )

        document.hybrid_score = (
            BM25_WEIGHT
            * bm25_normalized
            + SEMANTIC_WEIGHT
            * semantic
            + QUALITY_WEIGHT
            * quality_score(
                document.quality
            )
            + TEMPORAL_WEIGHT
            * temporal
        )

    ranked = sorted(
        documents,
        key=lambda document: (
            document.hybrid_score
        ),
        reverse=True,
    )

    return ranked[
        :top_k
    ]
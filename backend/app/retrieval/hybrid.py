from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.bm25 import retrieve_bm25
from backend.app.retrieval.dense import retrieve_dense


BM25_WEIGHT = 0.45
SEMANTIC_WEIGHT = 0.45
QUALITY_WEIGHT = 0.10


def normalize_scores(
    values: list[float],
) -> list[float]:
    if not values:
        return []

    minimum = min(values)
    maximum = max(values)

    if maximum == minimum:
        return [1.0 for _ in values]

    return [
        (value - minimum) / (maximum - minimum)
        for value in values
    ]


def quality_score(quality: str) -> float:
    scores = {
        "high": 1.0,
        "medium": 0.5,
        "low": 0.0,
    }

    return scores[quality]


def retrieve_hybrid(
    query: str,
    documents: list[RetrievedEvidence],
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    if not documents:
        return []

    retrieve_bm25(
        query=query,
        documents=documents,
        top_k=len(documents),
    )

    retrieve_dense(
        query=query,
        documents=documents,
        top_k=len(documents),
    )

    bm25_values = [
        document.bm25_score
        for document in documents
    ]

    normalized_bm25 = normalize_scores(
        bm25_values
    )

    for document, bm25_normalized in zip(
        documents,
        normalized_bm25,
    ):
        semantic = max(
            0.0,
            min(1.0, document.semantic_score),
        )

        document.hybrid_score = (
            BM25_WEIGHT * bm25_normalized
            + SEMANTIC_WEIGHT * semantic
            + QUALITY_WEIGHT
            * quality_score(document.quality)
        )

    ranked = sorted(
        documents,
        key=lambda document: document.hybrid_score,
        reverse=True,
    )

    return ranked[:top_k]
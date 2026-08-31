import re

from rank_bm25 import BM25Okapi

from backend.app.evidence.models import RetrievedEvidence


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def quality_bonus(quality: str) -> float:
    bonuses = {
        "high": 2.0,
        "medium": 1.0,
        "low": 0.0,
    }

    return bonuses[quality]


def retrieve_bm25(
    query: str,
    documents: list[RetrievedEvidence],
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    if not documents:
        return []

    tokenized_documents = [
        tokenize(document.text)
        for document in documents
    ]

    bm25 = BM25Okapi(tokenized_documents)

    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1] + quality_bonus(item[0].quality),
        reverse=True,
    )

    results = []

    for document, bm25_score in ranked[:top_k]:
        document.bm25_score = float(bm25_score)
        document.quality_bonus_score = quality_bonus(document.quality)
        document.retrieval_score = (
            document.bm25_score+document.quality_bonus_score
        )
        results.append(document)

    return results
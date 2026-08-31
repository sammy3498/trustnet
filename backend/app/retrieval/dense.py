from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.embeddings import embed_text, embed_texts
from backend.app.retrieval.excerpt import get_relevant_excerpt


def retrieve_dense(
    query: str,
    documents: list[RetrievedEvidence],
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    if not documents:
        return []

    texts = [
        get_relevant_excerpt(
            text=document.text,
            query=query,
            max_chars=2000,
        )
        for document in documents
    ]

    query_embedding = embed_text(query)
    document_embeddings = embed_texts(texts)

    scores = document_embeddings @ query_embedding

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: float(item[1]),
        reverse=True,
    )

    results = []

    for document, score in ranked[:top_k]:
        document.semantic_score = float(score)
        results.append(document)

    return results
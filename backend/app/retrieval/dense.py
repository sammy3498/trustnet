from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.embeddings import (
    embed_text,
    embed_texts,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


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


def retrieve_dense(
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

    texts = [
        get_document_passage(
            document=document,
            query=query,
            passages_by_url=(
                passages_by_url
            ),
        )
        for document in documents
    ]

    query_embedding = (
        embed_text(
            query
        )
    )

    document_embeddings = (
        embed_texts(
            texts
        )
    )

    scores = (
        document_embeddings
        @ query_embedding
    )

    ranked = sorted(
        zip(
            documents,
            scores,
        ),
        key=lambda item: (
            float(
                item[1]
            )
        ),
        reverse=True,
    )

    results = []

    for (
        document,
        score,
    ) in ranked[
        :top_k
    ]:
        document.semantic_score = (
            float(
                score
            )
        )

        results.append(
            document
        )

    return results
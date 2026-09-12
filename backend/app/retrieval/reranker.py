import os

from dotenv import (
    load_dotenv,
)
from sentence_transformers import (
    CrossEncoder,
)

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


load_dotenv()


DEFAULT_RERANKER_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)

_model = None


def get_reranker_model() -> (
    CrossEncoder
):
    global _model

    if _model is None:
        model_name = os.getenv(
            "RERANKER_MODEL",
            DEFAULT_RERANKER_MODEL,
        )

        _model = CrossEncoder(
            model_name,
            local_files_only=True,
        )

    return _model


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


def score_documents_for_query(
    query: str,
    documents: list[
        RetrievedEvidence
    ],
    passages_by_url: (
        dict[str, str] | None
    ) = None,
) -> list[
    tuple[
        RetrievedEvidence,
        float,
    ]
]:
    if not documents:
        return []

    model = (
        get_reranker_model()
    )

    pairs = [
        [
            query,
            get_document_passage(
                document=document,
                query=query,
                passages_by_url=(
                    passages_by_url
                ),
            ),
        ]
        for document
        in documents
    ]

    scores = model.predict(
        pairs
    )

    ranked = sorted(
        [
            (
                document,
                float(
                    score
                ),
            )
            for (
                document,
                score,
            )
            in zip(
                documents,
                scores,
            )
        ],
        key=lambda item: (
            item[1]
        ),
        reverse=True,
    )

    return ranked


def rerank_documents(
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
    ranked = (
        score_documents_for_query(
            query=query,
            documents=documents,
            passages_by_url=(
                passages_by_url
            ),
        )
    )

    results = []

    for (
        document,
        score,
    ) in ranked[
        :top_k
    ]:
        document.reranker_score = (
            score
        )

        results.append(
            document
        )

    return results
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
            model_name
        )

    return _model


def score_documents_for_query(
    query: str,
    documents: list[
        RetrievedEvidence
    ],
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
            get_relevant_excerpt(
                text=document.text,
                query=query,
                max_chars=(
                    DEFAULT_PASSAGE_MAX_CHARS
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
) -> list[
    RetrievedEvidence
]:
    ranked = (
        score_documents_for_query(
            query=query,
            documents=documents,
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
import os

from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
)


load_dotenv()


DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

_model = None


def get_embedding_model() -> (
    SentenceTransformer
):
    global _model

    if _model is None:
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            DEFAULT_EMBEDDING_MODEL,
        )

        _model = SentenceTransformer(
            model_name,
            local_files_only=True,
        )

    return _model


def embed_text(
    text: str,
):
    model = get_embedding_model()

    return model.encode(
        text,
        normalize_embeddings=True,
    )


def embed_texts(
    texts: list[str],
):
    model = get_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
    )
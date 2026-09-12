import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.retrieval.embeddings import (
    get_embedding_model,
)
from backend.app.retrieval.reranker import (
    get_reranker_model,
)


logger = logging.getLogger(
    __name__
)


def warm_retrieval_models() -> None:
    start = time.perf_counter()

    embedding_model = (
        get_embedding_model()
    )

    embedding_model.encode(
        [
            "TrustNet retrieval warmup.",
            "Evidence retrieval warmup.",
        ],
        normalize_embeddings=True,
    )

    reranker_model = (
        get_reranker_model()
    )

    reranker_model.predict(
        [
            [
                "TrustNet warmup query",
                "TrustNet warmup evidence",
            ]
        ]
    )

    elapsed_ms = round(
        (
            time.perf_counter()
            - start
        )
        * 1000
    )

    logger.warning(
        "TrustNet retrieval models "
        "warmed in %sms",
        elapsed_ms,
    )


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    warm_retrieval_models()

    yield


app = FastAPI(
    title="TrustNet API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
    ],
    allow_headers=[
        "*",
    ],
)


@app.get(
    "/health"
)
def health():
    return {
        "status": "ok"
    }


app.include_router(
    router
)
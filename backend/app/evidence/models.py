from typing import Literal

from pydantic import BaseModel


class RetrievedEvidence(BaseModel):
    title: str
    url: str
    source: str
    text: str
    quality: Literal[
        "high",
        "medium",
        "low",
    ]
    retrieval_score: float = 0.0
    bm25_score: float = 0.0
    quality_bonus_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    reranker_score: float = 0.0
    evidence_origin: Literal[
        "webpage",
        "snippet",
    ] = "webpage"
    published_date: str | None = None
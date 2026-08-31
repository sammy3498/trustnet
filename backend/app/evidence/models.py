from pydantic import BaseModel
from typing import Literal


class RetrievedEvidence(BaseModel):
    title: str
    url: str
    source: str
    text: str
    quality: Literal["high", "medium", "low"]
    retrieval_score: float = 0.0
    bm25_score: float = 0.0
    quality_bonus_score: float = 0.0
    retrieval_score: float = 0.0
    semantic_score: float = 0.0
    hybrid_score: float = 0.0
    reranker_score: float = 0.0
    evidence_origin: Literal["webpage", "snippet"] = "webpage"
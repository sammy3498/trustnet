from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.bm25 import retrieve_bm25
from backend.app.retrieval.candidate_builder import build_evidence_candidates


def retrieve_evidence(
    query: str,
    urls: list[str],
    top_k: int = 5,
) -> list[RetrievedEvidence]:
    candidates = build_evidence_candidates(urls)

    if not candidates:
        return []

    return retrieve_bm25(
        query=query,
        documents=candidates,
        top_k=top_k,
    )
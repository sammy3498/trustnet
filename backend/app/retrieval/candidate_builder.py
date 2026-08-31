from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.evidence_builder import build_evidence_from_url


def build_evidence_candidates(
    urls: list[str],
) -> list[RetrievedEvidence]:
    evidence = []

    for url in urls:
        try:
            item = build_evidence_from_url(url)
            evidence.append(item)
        except Exception:
            continue

    return evidence
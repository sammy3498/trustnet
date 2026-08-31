from backend.app.evidence.domain import get_publisher_domain
from backend.app.evidence.models import RetrievedEvidence


def select_diverse_sources(
    documents: list[RetrievedEvidence],
    top_k: int,
) -> list[RetrievedEvidence]:
    results = []
    seen_publishers = set()

    for item in documents:
        publisher = get_publisher_domain(item.source)

        if publisher in seen_publishers:
            continue

        seen_publishers.add(publisher)
        results.append(item)

        if len(results) == top_k:
            break

    return results
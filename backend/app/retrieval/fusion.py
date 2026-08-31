from backend.app.evidence.models import RetrievedEvidence


HYBRID_WEIGHT = 0.70
RERANKER_WEIGHT = 0.30
RRF_K = 60


def fuse_rankings(
    hybrid_ranked: list[RetrievedEvidence],
    reranked: list[RetrievedEvidence],
) -> list[RetrievedEvidence]:
    hybrid_positions = {
        item.url: rank
        for rank, item in enumerate(
            hybrid_ranked,
            start=1,
        )
    }

    reranker_positions = {
        item.url: rank
        for rank, item in enumerate(
            reranked,
            start=1,
        )
    }

    scores = {}

    for item in hybrid_ranked:
        hybrid_rank = hybrid_positions[item.url]
        reranker_rank = reranker_positions.get(
            item.url,
            len(reranked) + 1,
        )

        score = (
            HYBRID_WEIGHT
            / (RRF_K + hybrid_rank)
            + RERANKER_WEIGHT
            / (RRF_K + reranker_rank)
        )

        scores[item.url] = score

    return sorted(
        hybrid_ranked,
        key=lambda item: scores[item.url],
        reverse=True,
    )
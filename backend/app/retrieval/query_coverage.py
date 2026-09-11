import os

from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.reranker import (
    score_documents_for_query,
)
from backend.app.retrieval.temporal import (
    get_soft_temporal_fit_score,
)


DEFAULT_MIN_QUERY_RELEVANCE_SCORE = 0.0

RECOVERY_MAX_SEARCH_RANK = 2
SNIPPET_RECOVERY_MAX_SEARCH_RANK = 1

TEMPORAL_QUERY_BONUS_WEIGHT = 3.0

RELIABLE_QUALITIES = {
    "high",
    "medium",
}


def get_min_query_relevance_score() -> float:
    value = os.getenv(
        "QUERY_COVERAGE_MIN_RERANK_SCORE"
    )

    if value is None:
        return (
            DEFAULT_MIN_QUERY_RELEVANCE_SCORE
        )

    try:
        return float(
            value
        )

    except ValueError:
        return (
            DEFAULT_MIN_QUERY_RELEVANCE_SCORE
        )


def get_temporally_adjusted_score(
    raw_score: float,
    document: RetrievedEvidence,
    reference_text: str,
) -> float:
    temporal_fit = (
        get_soft_temporal_fit_score(
            reference_text=(
                reference_text
            ),
            title=document.title,
            text=document.text,
            published_date=(
                document.published_date
            ),
        )
    )

    temporal_adjustment = (
        TEMPORAL_QUERY_BONUS_WEIGHT
        * (
            temporal_fit - 0.5
        )
    )

    return (
        raw_score
        + temporal_adjustment
    )


def select_query_aware_evidence(
    primary_ranked: list[
        RetrievedEvidence
    ],
    fallback_ranked: list[
        RetrievedEvidence
    ],
    verification_queries: (
        list[str] | None
    ),
    url_query_ranks: dict[
        str,
        dict[str, int],
    ],
    top_k: int,
    temporal_reference_text: (
        str | None
    ) = None,
) -> list[
    RetrievedEvidence
]:
    if top_k <= 0:
        return []

    selected = []
    seen_urls = set()
    seen_publishers = set()

    minimum_relevance = (
        get_min_query_relevance_score()
    )

    def get_publisher(
        document: RetrievedEvidence,
    ) -> str:
        value = (
            document.source
            or document.url
        )

        return get_publisher_domain(
            value
        )

    def add_document(
        document: RetrievedEvidence,
        enforce_publisher_diversity: bool,
    ) -> bool:
        if (
            document.url
            in seen_urls
        ):
            return False

        publisher = (
            get_publisher(
                document
            )
        )

        if (
            enforce_publisher_diversity
            and publisher
            and publisher
            in seen_publishers
        ):
            return False

        selected.append(
            document
        )

        seen_urls.add(
            document.url
        )

        if publisher:
            seen_publishers.add(
                publisher
            )

        return True

    def try_add_candidates(
        candidates,
    ) -> bool:
        for (
            _,
            __,
            document,
        ) in candidates:
            if add_document(
                document,
                enforce_publisher_diversity=True,
            ):
                return True

        for (
            _,
            __,
            document,
        ) in candidates:
            if add_document(
                document,
                enforce_publisher_diversity=False,
            ):
                return True

        return False

    for verification_query in (
        verification_queries
        or []
    ):
        candidate_documents = []
        search_ranks = {}

        for document in (
            fallback_ranked
        ):
            query_ranks = (
                url_query_ranks.get(
                    document.url,
                    {},
                )
            )

            search_rank = (
                query_ranks.get(
                    verification_query
                )
            )

            if search_rank is None:
                continue

            candidate_documents.append(
                document
            )

            search_ranks[
                document.url
            ] = search_rank

        if not candidate_documents:
            continue

        scored = (
            score_documents_for_query(
                verification_query,
                candidate_documents,
            )
        )

        reference_text = (
            verification_query
            or temporal_reference_text
            or ""
        )

        all_candidates = [
            (
                get_temporally_adjusted_score(
                    raw_score=score,
                    document=document,
                    reference_text=(
                        reference_text
                    ),
                ),
                search_ranks[
                    document.url
                ],
                document,
            )
            for document, score
            in scored
        ]

        all_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        qualified_candidates = [
            item
            for item
            in all_candidates
            if item[0]
            >= minimum_relevance
        ]

        if qualified_candidates:
            try_add_candidates(
                qualified_candidates
            )

        elif all_candidates:
            best_candidate = (
                all_candidates[0]
            )

            (
                _,
                search_rank,
                document,
            ) = best_candidate

            reliable_recovery = (
                search_rank
                <= RECOVERY_MAX_SEARCH_RANK
                and document.quality
                in RELIABLE_QUALITIES
            )

            snippet_recovery = (
                document.evidence_origin
                == "snippet"
                and search_rank
                <= (
                    SNIPPET_RECOVERY_MAX_SEARCH_RANK
                )
            )

            if (
                reliable_recovery
                or snippet_recovery
            ):
                try_add_candidates(
                    [
                        best_candidate
                    ]
                )

        if (
            len(
                selected
            )
            >= top_k
        ):
            return selected[
                :top_k
            ]

    for document in (
        primary_ranked
    ):
        if (
            len(
                selected
            )
            >= top_k
        ):
            break

        add_document(
            document,
            enforce_publisher_diversity=True,
        )

    for document in (
        fallback_ranked
    ):
        if (
            len(
                selected
            )
            >= top_k
        ):
            break

        add_document(
            document,
            enforce_publisher_diversity=True,
        )

    if (
        len(
            selected
        )
        < top_k
    ):
        for document in (
            primary_ranked
            + fallback_ranked
        ):
            if (
                len(
                    selected
                )
                >= top_k
            ):
                break

            add_document(
                document,
                enforce_publisher_diversity=False,
            )

    return selected[
        :top_k
    ]
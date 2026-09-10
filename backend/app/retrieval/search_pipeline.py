import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval.evidence_builder import (
    build_evidence_from_snippet,
    build_evidence_from_url,
)
from backend.app.retrieval.fusion import (
    fuse_rankings,
)
from backend.app.retrieval.hybrid import (
    retrieve_hybrid,
)
from backend.app.retrieval.query_coverage import (
    select_query_aware_evidence,
)
from backend.app.retrieval.reranker import (
    rerank_documents,
)
from backend.app.retrieval.tavily_provider import (
    TavilySearchProvider,
)
from backend.app.retrieval.temporal import (
    is_explicitly_after_cutoff,
    is_publication_date_after_cutoff,
    is_text_explicitly_after_cutoff,
)


logger = logging.getLogger(
    __name__
)


def build_search_queries(
    query: str,
    alternate_query: str | None = None,
    verification_queries: list[str] | None = None,
    max_queries: int = 4,
) -> list[str]:
    queries = []
    candidates = []

    if alternate_query:
        candidates.append(
            alternate_query
        )

    candidates.extend(
        verification_queries or []
    )

    candidates.append(
        query
    )

    for candidate in candidates:
        if not candidate:
            continue

        candidate = (
            candidate.strip()
        )

        if (
            candidate
            and candidate not in queries
        ):
            queries.append(
                candidate
            )

    return queries[
        :max_queries
    ]


def is_excluded_search_result(
    url: str,
    title: str,
    excluded_patterns: (
        list[str] | None
    ) = None,
) -> bool:
    if not excluded_patterns:
        return False

    combined = (
        f"{url} {title}"
    ).lower()

    for pattern in excluded_patterns:
        normalized = (
            pattern.strip().lower()
        )

        if (
            normalized
            and normalized in combined
        ):
            return True

    return False


def run_search(
    query: str,
    max_results: int,
    end_date: str | None = None,
):
    provider = (
        TavilySearchProvider()
    )

    results = provider.search(
        query=query,
        max_results=max_results,
        end_date=end_date,
    )

    if (
        end_date
        and not results
    ):
        logger.warning(
            "Search returned no results "
            "with cutoff %s. Retrying "
            "without provider cutoff.",
            end_date,
        )

        results = provider.search(
            query=query,
            max_results=max_results,
            end_date=None,
        )

    return results


def search_and_retrieve(
    query: str,
    search_results: int = 10,
    top_k: int = 5,
    alternate_query: str | None = None,
    verification_queries: list[str] | None = None,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
) -> list[RetrievedEvidence]:
    queries = build_search_queries(
        query=query,
        alternate_query=alternate_query,
        verification_queries=verification_queries,
    )

    if not queries:
        return []

    search_result_groups = []

    with ThreadPoolExecutor(
        max_workers=min(
            4,
            len(queries),
        )
    ) as executor:
        futures = {
            executor.submit(
                run_search,
                search_query,
                search_results,
                end_date,
            ): search_query
            for search_query in queries
        }

        for future in as_completed(
            futures
        ):
            search_query = futures[
                future
            ]

            try:
                results = (
                    future.result()
                )

                search_result_groups.append(
                    (
                        search_query,
                        results,
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Search failed for query "
                    "'%s': %s",
                    search_query,
                    exc,
                )

    seen_urls = set()
    search_items = []
    url_query_ranks = {}

    for (
        search_query,
        results,
    ) in search_result_groups:
        for rank, result in enumerate(
            results,
            start=1,
        ):
            if not result.url:
                continue

            if is_excluded_search_result(
                url=result.url,
                title=result.title,
                excluded_patterns=(
                    excluded_result_patterns
                ),
            ):
                logger.warning(
                    "Skipping excluded "
                    "search result: %s",
                    result.url,
                )
                continue

            if is_explicitly_after_cutoff(
                title=result.title,
                url=result.url,
                end_date=end_date,
            ):
                logger.warning(
                    "Skipping explicitly "
                    "future-dated result: %s",
                    result.url,
                )
                continue

            query_ranks = (
                url_query_ranks.setdefault(
                    result.url,
                    {},
                )
            )

            previous_rank = (
                query_ranks.get(
                    search_query
                )
            )

            if (
                previous_rank is None
                or rank < previous_rank
            ):
                query_ranks[
                    search_query
                ] = rank

            if (
                result.url
                in seen_urls
            ):
                continue

            seen_urls.add(
                result.url
            )

            search_items.append(
                result
            )

    if not search_items:
        return []

    candidates = []

    with ThreadPoolExecutor(
        max_workers=5
    ) as executor:
        futures = {
            executor.submit(
                build_evidence_from_url,
                result.url,
                result.title,
            ): result
            for result in search_items
        }

        for future in as_completed(
            futures
        ):
            result = futures[
                future
            ]

            try:
                evidence = (
                    future.result()
                )

            except Exception as exc:
                logger.warning(
                    "Failed to build "
                    "evidence from %s: %s",
                    result.url,
                    exc,
                )

                if not (
                    result.snippet.strip()
                ):
                    continue

                evidence = (
                    build_evidence_from_snippet(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                    )
                )

            if (
                evidence.evidence_origin
                == "webpage"
                and is_publication_date_after_cutoff(
                    published_date=(
                        evidence.published_date
                    ),
                    end_date=end_date,
                )
            ):
                logger.warning(
                    "Skipping post-cutoff "
                    "publication %s dated %s",
                    evidence.url,
                    evidence.published_date,
                )
                continue

            if (
                evidence.evidence_origin
                == "webpage"
                and not evidence.published_date
                and is_text_explicitly_after_cutoff(
                    text=evidence.text,
                    end_date=end_date,
                )
            ):
                logger.warning(
                    "Skipping evidence with "
                    "post-cutoff publication "
                    "or update date: %s",
                    evidence.url,
                )
                continue

            candidates.append(
                evidence
            )

    if not candidates:
        return []

    hybrid_ranked = (
        retrieve_hybrid(
            query=query,
            documents=candidates,
            top_k=len(
                candidates
            ),
        )
    )

    rerank_count = min(
        max(
            top_k * 2,
            6,
        ),
        len(
            hybrid_ranked
        ),
    )

    rerank_candidates = (
        hybrid_ranked[
            :rerank_count
        ]
    )

    reranked = (
        rerank_documents(
            query=query,
            documents=(
                rerank_candidates
            ),
            top_k=len(
                rerank_candidates
            ),
        )
    )

    fused_ranked = (
        fuse_rankings(
            hybrid_ranked=(
                rerank_candidates
            ),
            reranked=reranked,
        )
    )

    return select_query_aware_evidence(
        primary_ranked=(
            fused_ranked
        ),
        fallback_ranked=(
            hybrid_ranked
        ),
        verification_queries=(
            verification_queries
        ),
        url_query_ranks=(
            url_query_ranks
        ),
        top_k=top_k,
    )
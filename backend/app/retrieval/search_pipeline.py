import logging
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
    as_completed,
)

from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.evidence.provenance import (
    classify_source_quality,
)
from backend.app.retrieval.evidence_builder import (
    build_evidence_from_snippet,
    build_evidence_from_url,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
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

MAX_SEARCH_QUERIES = 4

MAX_TAVILY_SEARCH_CALLS = 2

MIN_RELIABLE_SEARCH_PUBLISHERS = 2

MAX_FETCH_CANDIDATES = 5
MAX_FETCH_WORKERS = 5
MAX_TOTAL_CANDIDATES = 10

SEARCH_STAGE_TIMEOUT_SECONDS = 4.0
FETCH_STAGE_TIMEOUT_SECONDS = 3.25

RERANK_SKIP_AFTER_SECONDS = 8.0

QUALITY_PRIORITY = {
    "high": 0,
    "medium": 1,
    "low": 2,
}


def build_search_queries(
    query: str,
    alternate_query: str | None = None,
    verification_queries: (
        list[str] | None
    ) = None,
    max_queries: int = (
        MAX_SEARCH_QUERIES
    ),
) -> list[str]:
    queries = []
    candidates = []

    if alternate_query:
        candidates.append(
            alternate_query
        )

    candidates.extend(
        verification_queries
        or []
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
            and candidate
            not in queries
        ):
            queries.append(
                candidate
            )

    return queries[
        :max_queries
    ]


def build_authority_search_query(
    query: str,
) -> str | None:
    normalized = query.strip()

    if not normalized:
        return None

    return (
        f"{normalized} "
        "official primary source"
    )


def add_authority_search_query(
    queries: list[str],
    query: str,
    max_queries: int = (
        MAX_SEARCH_QUERIES
    ),
) -> list[str]:
    result = list(
        queries
    )

    if (
        len(result)
        >= max_queries
    ):
        return result[
            :max_queries
        ]

    authority_query = (
        build_authority_search_query(
            query
        )
    )

    if (
        authority_query
        and authority_query
        not in result
    ):
        result.append(
            authority_query
        )

    return result[
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

    for pattern in (
        excluded_patterns
    ):
        normalized = (
            pattern
            .strip()
            .lower()
        )

        if (
            normalized
            and normalized
            in combined
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

    return provider.search(
        query=query,
        max_results=max_results,
        end_date=end_date,
    )


def is_search_result_usable(
    result,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
) -> bool:
    if not result.url:
        return False

    if is_excluded_search_result(
        url=result.url,
        title=result.title,
        excluded_patterns=(
            excluded_result_patterns
        ),
    ):
        return False

    if is_explicitly_after_cutoff(
        title=result.title,
        url=result.url,
        end_date=end_date,
    ):
        return False

    return True


def get_reliable_search_publishers(
    results,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
) -> set[str]:
    publishers = set()

    for result in results:
        if not (
            is_search_result_usable(
                result=result,
                end_date=end_date,
                excluded_result_patterns=(
                    excluded_result_patterns
                ),
            )
        ):
            continue

        quality = (
            classify_source_quality(
                result.url
            )
        )

        if quality not in {
            "high",
            "medium",
        }:
            continue

        publisher = (
            get_publisher_domain(
                result.url
            )
        )

        if publisher:
            publishers.add(
                publisher
            )

    return publishers


def has_sufficient_reliable_search_results(
    results,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
    min_publishers: int = (
        MIN_RELIABLE_SEARCH_PUBLISHERS
    ),
) -> bool:
    required = max(
        1,
        int(
            min_publishers
        ),
    )

    publishers = (
        get_reliable_search_publishers(
            results=results,
            end_date=end_date,
            excluded_result_patterns=(
                excluded_result_patterns
            ),
        )
    )

    return (
        len(
            publishers
        )
        >= required
    )


def build_adaptive_search_plan(
    query: str,
    candidate_queries: list[str],
) -> list[str]:
    if not candidate_queries:
        return []

    primary_query = (
        candidate_queries[0]
    )

    plan = [
        primary_query
    ]

    authority_query = (
        build_authority_search_query(
            query
        )
    )

    if (
        authority_query
        and authority_query
        != primary_query
    ):
        plan.append(
            authority_query
        )

    if (
        len(plan)
        < MAX_TAVILY_SEARCH_CALLS
    ):
        for candidate in (
            candidate_queries[1:]
        ):
            if (
                candidate
                and candidate
                not in plan
            ):
                plan.append(
                    candidate
                )

            if (
                len(plan)
                >= MAX_TAVILY_SEARCH_CALLS
            ):
                break

    return plan[
        :MAX_TAVILY_SEARCH_CALLS
    ]


def run_adaptive_searches(
    query: str,
    candidate_queries: list[str],
    search_results: int,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
    min_reliable_publishers: int = (
        MIN_RELIABLE_SEARCH_PUBLISHERS
    ),
):
    search_plan = (
        build_adaptive_search_plan(
            query=query,
            candidate_queries=(
                candidate_queries
            ),
        )
    )

    if not search_plan:
        return {}, []

    search_results_by_query = {}
    executed_queries = []

    search_start = (
        time.perf_counter()
    )

    search_executor = (
        ThreadPoolExecutor(
            max_workers=1
        )
    )

    try:
        for index, search_query in (
            enumerate(
                search_plan
            )
        ):
            if (
                len(
                    executed_queries
                )
                >= MAX_TAVILY_SEARCH_CALLS
            ):
                break

            elapsed = (
                time.perf_counter()
                - search_start
            )

            remaining = (
                SEARCH_STAGE_TIMEOUT_SECONDS
                - elapsed
            )

            if remaining <= 0:
                logger.warning(
                    "Search stage exhausted "
                    "its %.1f-second budget.",
                    SEARCH_STAGE_TIMEOUT_SECONDS,
                )

                break

            future = (
                search_executor.submit(
                    run_search,
                    search_query,
                    search_results,
                    end_date,
                )
            )

            executed_queries.append(
                search_query
            )

            try:
                results = (
                    future.result(
                        timeout=remaining
                    )
                )

            except FuturesTimeoutError:
                logger.warning(
                    "Search query '%s' exceeded "
                    "the remaining retrieval "
                    "search budget.",
                    search_query,
                )

                future.cancel()

                break

            except Exception as exc:
                logger.warning(
                    "Search failed for query "
                    "'%s': %s",
                    search_query,
                    exc,
                )

                results = []

            search_results_by_query[
                search_query
            ] = results

            if (
                index == 0
                and has_sufficient_reliable_search_results(
                    results=results,
                    end_date=end_date,
                    excluded_result_patterns=(
                        excluded_result_patterns
                    ),
                    min_publishers=(
                        min_reliable_publishers
                    ),
                )
            ):
                logger.warning(
                    "Primary search produced "
                    "enough independent "
                    "reliable publishers. "
                    "Skipping backup Tavily "
                    "search."
                )

                break

    finally:
        search_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    return (
        search_results_by_query,
        executed_queries,
    )


def get_search_result_metadata(
    result,
    url_query_ranks: dict[
        str,
        dict[str, int],
    ],
):
    quality = (
        classify_source_quality(
            result.url
        )
    )

    query_ranks = (
        url_query_ranks.get(
            result.url,
            {},
        )
    )

    if query_ranks:
        best_rank = min(
            query_ranks.values()
        )

        query_hits = len(
            query_ranks
        )

    else:
        best_rank = 999
        query_hits = 0

    if best_rank <= 3:
        relevance_band = 0

    elif best_rank <= 5:
        relevance_band = 1

    else:
        relevance_band = 2

    return {
        "quality": quality,
        "best_rank": best_rank,
        "query_hits": query_hits,
        "relevance_band": (
            relevance_band
        ),
    }


def get_search_item_priority(
    result,
    url_query_ranks: dict[
        str,
        dict[str, int],
    ],
):
    metadata = (
        get_search_result_metadata(
            result,
            url_query_ranks,
        )
    )

    return (
        metadata[
            "relevance_band"
        ],
        QUALITY_PRIORITY.get(
            metadata[
                "quality"
            ],
            3,
        ),
        metadata[
            "best_rank"
        ],
        -metadata[
            "query_hits"
        ],
        result.url,
    )


def select_fetch_candidates(
    search_items,
    url_query_ranks: dict[
        str,
        dict[str, int],
    ],
    max_candidates: int = (
        MAX_FETCH_CANDIDATES
    ),
):
    if (
        max_candidates <= 0
        or not search_items
    ):
        return []

    prioritized = sorted(
        search_items,
        key=lambda item: (
            get_search_item_priority(
                item,
                url_query_ranks,
            )
        ),
    )

    selected = []
    selected_urls = set()
    selected_publishers = set()

    def add_item(
        item,
        enforce_diversity: bool,
    ) -> bool:
        if (
            item.url
            in selected_urls
        ):
            return False

        publisher = (
            get_publisher_domain(
                item.url
            )
        )

        if (
            enforce_diversity
            and publisher
            and publisher
            in selected_publishers
        ):
            return False

        selected.append(
            item
        )

        selected_urls.add(
            item.url
        )

        if publisher:
            selected_publishers.add(
                publisher
            )

        return True

    for preferred_quality in (
        "high",
        "medium",
    ):
        for item in prioritized:
            if (
                len(selected)
                >= max_candidates
            ):
                return selected

            metadata = (
                get_search_result_metadata(
                    item,
                    url_query_ranks,
                )
            )

            if (
                metadata[
                    "quality"
                ]
                != preferred_quality
            ):
                continue

            add_item(
                item,
                enforce_diversity=True,
            )

    for item in prioritized:
        if (
            len(selected)
            >= max_candidates
        ):
            return selected

        add_item(
            item,
            enforce_diversity=True,
        )

    for item in prioritized:
        if (
            len(selected)
            >= max_candidates
        ):
            break

        add_item(
            item,
            enforce_diversity=False,
        )

    return selected


def build_snippet_fallback(
    result,
) -> RetrievedEvidence | None:
    snippet = (
        result.snippet
        or ""
    ).strip()

    if not snippet:
        return None

    return (
        build_evidence_from_snippet(
            url=result.url,
            title=result.title,
            snippet=snippet,
        )
    )


def fetch_evidence_pages(
    fetch_items,
    end_date: str | None = None,
    timeout_seconds: float = (
        FETCH_STAGE_TIMEOUT_SECONDS
    ),
) -> dict[
    str,
    RetrievedEvidence,
]:
    if not fetch_items:
        return {}

    evidence_by_url = {}

    fetch_executor = (
        ThreadPoolExecutor(
            max_workers=min(
                MAX_FETCH_WORKERS,
                len(
                    fetch_items
                ),
            )
        )
    )

    fetch_futures = {
        fetch_executor.submit(
            build_evidence_from_url,
            result.url,
            result.title,
        ): result
        for result
        in fetch_items
    }

    try:
        try:
            for future in (
                as_completed(
                    fetch_futures,
                    timeout=(
                        timeout_seconds
                    ),
                )
            ):
                result = (
                    fetch_futures[
                        future
                    ]
                )

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

                    continue

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
                    and not (
                        evidence.published_date
                    )
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

                evidence_by_url[
                    result.url
                ] = evidence

        except FuturesTimeoutError:
            logger.warning(
                "Page fetch stage exceeded "
                "%.2f seconds. Continuing "
                "with completed pages and "
                "search snippets.",
                timeout_seconds,
            )

    finally:
        for future in (
            fetch_futures
        ):
            if not future.done():
                future.cancel()

        fetch_executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    return evidence_by_url


def search_and_retrieve(
    query: str,
    search_results: int = 10,
    top_k: int = 5,
    alternate_query: str | None = None,
    verification_queries: (
        list[str] | None
    ) = None,
    end_date: str | None = None,
    excluded_result_patterns: (
        list[str] | None
    ) = None,
) -> list[
    RetrievedEvidence
]:
    retrieval_start = (
        time.perf_counter()
    )

    candidate_queries = (
        build_search_queries(
            query=query,
            alternate_query=(
                alternate_query
            ),
            verification_queries=(
                verification_queries
            ),
        )
    )

    if not candidate_queries:
        return []

    search_start = (
        time.perf_counter()
    )

    minimum_publishers = min(
        2,
        max(
            1,
            top_k,
        ),
    )

    (
        search_results_by_query,
        queries,
    ) = run_adaptive_searches(
        query=query,
        candidate_queries=(
            candidate_queries
        ),
        search_results=(
            search_results
        ),
        end_date=end_date,
        excluded_result_patterns=(
            excluded_result_patterns
        ),
        min_reliable_publishers=(
            minimum_publishers
        ),
    )

    search_end = (
        time.perf_counter()
    )

    if not queries:
        return []

    temporal_reference_text = (
        "\n".join(
            queries
        )
    )

    seen_urls = set()
    search_items = []
    url_query_ranks = {}

    for search_query in queries:
        results = (
            search_results_by_query.get(
                search_query,
                [],
            )
        )

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
                or rank
                < previous_rank
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

    prioritized_search_items = (
        sorted(
            search_items,
            key=lambda item: (
                get_search_item_priority(
                    item,
                    url_query_ranks,
                )
            ),
        )
    )

    fetch_items = (
        select_fetch_candidates(
            search_items=(
                search_items
            ),
            url_query_ranks=(
                url_query_ranks
            ),
            max_candidates=(
                MAX_FETCH_CANDIDATES
            ),
        )
    )

    fetch_start = (
        time.perf_counter()
    )

    evidence_by_url = (
        fetch_evidence_pages(
            fetch_items=(
                fetch_items
            ),
            end_date=end_date,
            timeout_seconds=(
                FETCH_STAGE_TIMEOUT_SECONDS
            ),
        )
    )

    candidates = []

    for result in fetch_items:
        evidence = (
            evidence_by_url.get(
                result.url
            )
        )

        if evidence is None:
            evidence = (
                build_snippet_fallback(
                    result
                )
            )

        if evidence is not None:
            candidates.append(
                evidence
            )

    if (
        len(
            candidates
        )
        < MAX_TOTAL_CANDIDATES
    ):
        fetched_urls = {
            item.url
            for item in fetch_items
        }

        for result in (
            prioritized_search_items
        ):
            if (
                len(
                    candidates
                )
                >= MAX_TOTAL_CANDIDATES
            ):
                break

            if (
                result.url
                in fetched_urls
            ):
                continue

            evidence = (
                build_snippet_fallback(
                    result
                )
            )

            if evidence is not None:
                candidates.append(
                    evidence
                )

    fetch_end = (
        time.perf_counter()
    )

    if not candidates:
        return []

    passage_start = (
        time.perf_counter()
    )

    passages_by_url = {
        document.url: (
            get_relevant_excerpt(
                text=document.text,
                query=query,
                max_chars=(
                    DEFAULT_PASSAGE_MAX_CHARS
                ),
            )
        )
        for document in candidates
    }

    passage_end = (
        time.perf_counter()
    )

    hybrid_start = (
        time.perf_counter()
    )

    hybrid_ranked = (
        retrieve_hybrid(
            query=query,
            documents=candidates,
            top_k=len(
                candidates
            ),
            temporal_reference_text=(
                temporal_reference_text
            ),
            passages_by_url=(
                passages_by_url
            ),
        )
    )

    hybrid_end = (
        time.perf_counter()
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

    rerank_start = (
        time.perf_counter()
    )

    elapsed_before_rerank = (
        rerank_start
        - retrieval_start
    )

    rerank_skipped = (
        elapsed_before_rerank
        >= RERANK_SKIP_AFTER_SECONDS
        or len(
            rerank_candidates
        )
        <= 1
    )

    if rerank_skipped:
        reranked = list(
            rerank_candidates
        )

        logger.warning(
            "Skipping cross-encoder "
            "reranking because retrieval "
            "has already consumed %.2f "
            "seconds.",
            elapsed_before_rerank,
        )

    else:
        reranked = (
            rerank_documents(
                query=query,
                documents=(
                    rerank_candidates
                ),
                top_k=len(
                    rerank_candidates
                ),
                passages_by_url=(
                    passages_by_url
                ),
            )
        )

    rerank_end = (
        time.perf_counter()
    )

    fusion_start = (
        time.perf_counter()
    )

    fused_ranked = (
        fuse_rankings(
            hybrid_ranked=(
                rerank_candidates
            ),
            reranked=reranked,
        )
    )

    fusion_end = (
        time.perf_counter()
    )

    selection_start = (
        time.perf_counter()
    )

    selected = (
        select_query_aware_evidence(
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
            temporal_reference_text=(
                temporal_reference_text
            ),
        )
    )

    selection_end = (
        time.perf_counter()
    )

    logger.warning(
        (
            "TrustNet retrieval timing | "
            "search=%sms | "
            "fetch=%sms | "
            "passages=%sms | "
            "hybrid=%sms | "
            "rerank=%sms | "
            "fusion=%sms | "
            "selection=%sms | "
            "total=%sms | "
            "queries=%s | "
            "urls=%s | "
            "fetch_targets=%s | "
            "webpages=%s | "
            "candidates=%s | "
            "selected=%s | "
            "rerank_skipped=%s"
        ),
        round(
            (
                search_end
                - search_start
            )
            * 1000
        ),
        round(
            (
                fetch_end
                - fetch_start
            )
            * 1000
        ),
        round(
            (
                passage_end
                - passage_start
            )
            * 1000
        ),
        round(
            (
                hybrid_end
                - hybrid_start
            )
            * 1000
        ),
        round(
            (
                rerank_end
                - rerank_start
            )
            * 1000
        ),
        round(
            (
                fusion_end
                - fusion_start
            )
            * 1000
        ),
        round(
            (
                selection_end
                - selection_start
            )
            * 1000
        ),
        round(
            (
                selection_end
                - retrieval_start
            )
            * 1000
        ),
        len(
            queries
        ),
        len(
            search_items
        ),
        len(
            fetch_items
        ),
        len(
            evidence_by_url
        ),
        len(
            candidates
        ),
        len(
            selected
        ),
        rerank_skipped,
    )

    return selected
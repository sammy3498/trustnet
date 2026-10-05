import logging
import time
from concurrent.futures import ALL_COMPLETED, ThreadPoolExecutor, wait

from backend.app.evidence.domain import get_publisher_domain
from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.evidence_builder import (
    build_evidence_from_snippet,
    build_evidence_from_url,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)
from backend.app.retrieval.fusion import fuse_rankings
from backend.app.retrieval.hybrid import retrieve_hybrid
from backend.app.retrieval.query_coverage import select_query_aware_evidence
from backend.app.retrieval.reranker import rerank_documents
from backend.app.retrieval.tavily_provider import TavilySearchProvider
from backend.app.retrieval.temporal import (
    is_explicitly_after_cutoff,
    is_publication_date_after_cutoff,
    is_text_explicitly_after_cutoff,
)
from backend.app.utils.config import get_mode_limits
from backend.app.utils.time_budget import AnalysisBudget


logger = logging.getLogger(__name__)


def build_search_queries(
    query: str,
    alternate_query: str | None = None,
    verification_queries: list[str] | None = None,
    max_queries: int = 4,
) -> list[str]:
    queries = []
    candidates = []

    if alternate_query:
        candidates.append(alternate_query)

    candidates.extend(verification_queries or [])
    candidates.append(query)

    for candidate in candidates:
        if not candidate:
            continue

        candidate = candidate.strip()

        if candidate and candidate not in queries:
            queries.append(candidate)

    return queries[:max_queries]


def is_excluded_search_result(
    url: str,
    title: str,
    excluded_patterns: list[str] | None = None,
) -> bool:
    if not excluded_patterns:
        return False

    combined = f"{url} {title}".lower()

    for pattern in excluded_patterns:
        normalized = pattern.strip().lower()
        if normalized and normalized in combined:
            return True

    return False


def run_search(
    query: str,
    max_results: int,
    end_date: str | None = None,
):
    provider = TavilySearchProvider()

    results = provider.search(
        query=query,
        max_results=max_results,
        end_date=end_date,
    )

    if end_date and not results:
        logger.warning(
            "Search returned no results with cutoff %s. Retrying without "
            "provider cutoff.",
            end_date,
        )
        results = provider.search(
            query=query,
            max_results=max_results,
            end_date=None,
        )

    return results


def _stage_timeout(
    budget: AnalysisBudget | None,
    configured_max_seconds: float,
    reserve_seconds: float = 0.0,
) -> float | None:
    if budget is None:
        return configured_max_seconds

    available = budget.remaining() - reserve_seconds
    if available <= 0:
        return 0.0

    return max(0.1, min(configured_max_seconds, available))


def _fast_select(
    ranked: list[RetrievedEvidence],
    fallback: list[RetrievedEvidence],
    top_k: int,
) -> list[RetrievedEvidence]:
    selected = []
    seen_urls = set()
    seen_publishers = set()

    for enforce_diversity in (True, False):
        for document in ranked + fallback:
            if len(selected) >= top_k:
                return selected

            if document.url in seen_urls:
                continue

            publisher = get_publisher_domain(
                document.source or document.url
            )

            if (
                enforce_diversity
                and publisher
                and publisher in seen_publishers
            ):
                continue

            selected.append(document)
            seen_urls.add(document.url)

            if publisher:
                seen_publishers.add(publisher)

    return selected[:top_k]


QUALITY_SELECTION_BONUS = {
    "high": 12.0,
    "medium": 6.0,
    "low": 0.0,
}


def _deep_quality_select(
    base_selected: list[RetrievedEvidence],
    ranked: list[RetrievedEvidence],
    fallback: list[RetrievedEvidence],
    top_k: int,
) -> list[RetrievedEvidence]:
    if top_k <= 0:
        return []

    pool_limit = max(
        top_k * 3,
        10,
    )

    pool = []
    seen_urls = set()

    for document in (
        ranked
        + fallback
        + base_selected
    ):
        if document.url in seen_urls:
            continue

        seen_urls.add(
            document.url
        )
        pool.append(
            document
        )

        if len(pool) >= pool_limit:
            break

    if not pool:
        return []

    ranked_positions = {
        document.url: index
        for index, document
        in enumerate(
            ranked,
            start=1,
        )
    }

    fallback_positions = {
        document.url: index
        for index, document
        in enumerate(
            fallback,
            start=1,
        )
    }

    base_urls = {
        document.url
        for document in base_selected
    }

    scored = []

    for document in pool:
        ranked_position = (
            ranked_positions.get(
                document.url,
                pool_limit + 1,
            )
        )

        fallback_position = (
            fallback_positions.get(
                document.url,
                pool_limit + 1,
            )
        )

        relevance_position = min(
            ranked_position,
            fallback_position,
        )

        relevance_score = max(
            0.0,
            100.0
            - (
                relevance_position
                - 1
            )
            * 6.0,
        )

        quality_bonus = (
            QUALITY_SELECTION_BONUS.get(
                document.quality,
                0.0,
            )
        )

        coverage_bonus = (
            3.0
            if document.url
            in base_urls
            else 0.0
        )

        total_score = (
            relevance_score
            + quality_bonus
            + coverage_bonus
        )

        scored.append(
            (
                total_score,
                -relevance_position,
                document,
            )
        )

    scored.sort(
        key=lambda item: (
            item[0],
            item[1],
        ),
        reverse=True,
    )

    ordered = [
        item[2]
        for item in scored
    ]

    selected = []
    selected_urls = set()
    selected_publishers = set()

    for enforce_diversity in (
        True,
        False,
    ):
        for document in ordered:
            if len(selected) >= top_k:
                break

            if document.url in selected_urls:
                continue

            publisher = get_publisher_domain(
                document.source
                or document.url
            )

            if (
                enforce_diversity
                and publisher
                and publisher
                in selected_publishers
            ):
                continue

            selected.append(
                document
            )
            selected_urls.add(
                document.url
            )

            if publisher:
                selected_publishers.add(
                    publisher
                )

        if len(selected) >= top_k:
            break

    return selected[:top_k]


def search_and_retrieve(
    query: str,
    search_results: int = 10,
    top_k: int = 5,
    alternate_query: str | None = None,
    verification_queries: list[str] | None = None,
    end_date: str | None = None,
    excluded_result_patterns: list[str] | None = None,
    budget: AnalysisBudget | None = None,
    mode: str = "normal",
) -> list[RetrievedEvidence]:
    retrieval_start = time.perf_counter()
    limits = get_mode_limits(mode)

    queries = build_search_queries(
        query=query,
        alternate_query=alternate_query,
        verification_queries=verification_queries,
        max_queries=limits["max_queries"],
    )

    if not queries:
        return []

    temporal_reference_text = "\n".join(queries)
    search_result_groups = []

    search_start = time.perf_counter()
    search_timeout = _stage_timeout(
        budget,
        limits["search_stage_max_seconds"],
        reserve_seconds=limits["finalization_reserve_seconds"] + 2.0,
    )

    if search_timeout == 0.0:
        return []

    search_executor = ThreadPoolExecutor(
        max_workers=min(4, len(queries))
    )
    search_futures = {
        search_executor.submit(
            run_search,
            search_query,
            search_results,
            end_date,
        ): search_query
        for search_query in queries
    }

    done, not_done = wait(
        search_futures,
        timeout=search_timeout,
        return_when=ALL_COMPLETED,
    )

    for future in done:
        search_query = search_futures[future]
        try:
            results = future.result()
            search_result_groups.append((search_query, results))
        except Exception as exc:
            logger.warning(
                "Search failed for query '%s': %s",
                search_query,
                exc,
            )

    for future in not_done:
        future.cancel()

    search_executor.shutdown(wait=False, cancel_futures=True)
    search_end = time.perf_counter()

    seen_urls = set()
    search_items = []
    url_query_ranks = {}

    for search_query, results in search_result_groups:
        for rank, result in enumerate(results, start=1):
            if not result.url:
                continue

            if is_excluded_search_result(
                url=result.url,
                title=result.title,
                excluded_patterns=excluded_result_patterns,
            ):
                logger.warning(
                    "Skipping excluded search result: %s",
                    result.url,
                )
                continue

            if is_explicitly_after_cutoff(
                title=result.title,
                url=result.url,
                end_date=end_date,
            ):
                logger.warning(
                    "Skipping explicitly future-dated result: %s",
                    result.url,
                )
                continue

            query_ranks = url_query_ranks.setdefault(result.url, {})
            previous_rank = query_ranks.get(search_query)

            if previous_rank is None or rank < previous_rank:
                query_ranks[search_query] = rank

            if result.url in seen_urls:
                continue

            seen_urls.add(result.url)
            search_items.append(result)

    if not search_items:
        return []

    candidates = []
    fetch_start = time.perf_counter()

    fetch_timeout = _stage_timeout(
        budget,
        limits["fetch_stage_max_seconds"],
        reserve_seconds=limits["finalization_reserve_seconds"] + 2.0,
    )

    if fetch_timeout == 0.0:
        return []

    fetch_executor = ThreadPoolExecutor(max_workers=5)
    fetch_futures = {
        fetch_executor.submit(
            build_evidence_from_url,
            result.url,
            result.title,
        ): result
        for result in search_items
    }

    done, not_done = wait(
        fetch_futures,
        timeout=fetch_timeout,
        return_when=ALL_COMPLETED,
    )

    completed_results = set()

    for future in done:
        result = fetch_futures[future]
        completed_results.add(result.url)

        try:
            evidence = future.result()
        except Exception as exc:
            logger.warning(
                "Failed to build evidence from %s: %s",
                result.url,
                exc,
            )

            if not result.snippet.strip():
                continue

            evidence = build_evidence_from_snippet(
                url=result.url,
                title=result.title,
                snippet=result.snippet,
            )

        if (
            evidence.evidence_origin == "webpage"
            and is_publication_date_after_cutoff(
                published_date=evidence.published_date,
                end_date=end_date,
            )
        ):
            logger.warning(
                "Skipping post-cutoff publication %s dated %s",
                evidence.url,
                evidence.published_date,
            )
            continue

        if (
            evidence.evidence_origin == "webpage"
            and not evidence.published_date
            and is_text_explicitly_after_cutoff(
                text=evidence.text,
                end_date=end_date,
            )
        ):
            logger.warning(
                "Skipping evidence with post-cutoff publication or update "
                "date: %s",
                evidence.url,
            )
            continue

        candidates.append(evidence)

    for future in not_done:
        result = fetch_futures[future]
        future.cancel()

        if result.url in completed_results or not result.snippet.strip():
            continue

        try:
            candidates.append(
                build_evidence_from_snippet(
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                )
            )
        except Exception as exc:
            logger.warning(
                "Failed to preserve timed-out fetch as snippet %s: %s",
                result.url,
                exc,
            )

    fetch_executor.shutdown(wait=False, cancel_futures=True)
    fetch_end = time.perf_counter()

    if not candidates:
        return []

    passage_start = time.perf_counter()
    passages_by_url = {
        document.url: get_relevant_excerpt(
            text=document.text,
            query=query,
            max_chars=DEFAULT_PASSAGE_MAX_CHARS,
        )
        for document in candidates
    }
    passage_end = time.perf_counter()

    hybrid_start = time.perf_counter()
    hybrid_ranked = retrieve_hybrid(
        query=query,
        documents=candidates,
        top_k=len(candidates),
        temporal_reference_text=temporal_reference_text,
        passages_by_url=passages_by_url,
    )
    hybrid_end = time.perf_counter()

    rerank_count = min(
        max(top_k * 2, 6),
        len(hybrid_ranked),
    )
    rerank_candidates = hybrid_ranked[:rerank_count]

    rerank_start = time.perf_counter()

    should_rerank = (
        not budget
        or (
            not budget.should_stop_optional_work()
            and budget.can_start(
                expected_seconds=1.5,
                reserve_seconds=limits["finalization_reserve_seconds"] + 1.0,
            )
        )
    )

    if should_rerank:
        reranked = rerank_documents(
            query=query,
            documents=rerank_candidates,
            top_k=len(rerank_candidates),
            passages_by_url=passages_by_url,
        )
        fused_ranked = fuse_rankings(
            hybrid_ranked=rerank_candidates,
            reranked=reranked,
        )
    else:
        logger.warning(
            "Skipping cross-encoder reranking because the analysis budget "
            "is close to its deadline."
        )
        reranked = rerank_candidates
        fused_ranked = rerank_candidates

    rerank_end = time.perf_counter()

    fusion_start = rerank_end
    fusion_end = time.perf_counter()

    selection_start = time.perf_counter()

    should_run_query_coverage = (
        verification_queries
        and (
            not budget
            or (
                not budget.should_stop_optional_work()
                and budget.can_start(
                    expected_seconds=1.5,
                    reserve_seconds=(
                        limits["finalization_reserve_seconds"] + 1.0
                    ),
                )
            )
        )
    )

    if should_run_query_coverage:
        selected = select_query_aware_evidence(
            primary_ranked=fused_ranked,
            fallback_ranked=hybrid_ranked,
            verification_queries=verification_queries,
            url_query_ranks=url_query_ranks,
            top_k=top_k,
            temporal_reference_text=temporal_reference_text,
        )
    else:
        selected = _fast_select(
            ranked=fused_ranked,
            fallback=hybrid_ranked,
            top_k=top_k,
        )

    if mode == "deep":
        selected = _deep_quality_select(
            base_selected=selected,
            ranked=fused_ranked,
            fallback=hybrid_ranked,
            top_k=top_k,
        )

    selection_end = time.perf_counter()

    logger.warning(
        (
            "TrustNet retrieval timing | mode=%s | search=%sms | "
            "fetch=%sms | passages=%sms | hybrid=%sms | rerank=%sms | "
            "fusion=%sms | selection=%sms | total=%sms | queries=%s | "
            "urls=%s | candidates=%s | selected=%s"
        ),
        mode,
        round((search_end - search_start) * 1000),
        round((fetch_end - fetch_start) * 1000),
        round((passage_end - passage_start) * 1000),
        round((hybrid_end - hybrid_start) * 1000),
        round((rerank_end - rerank_start) * 1000),
        round((fusion_end - fusion_start) * 1000),
        round((selection_end - selection_start) * 1000),
        round((selection_end - retrieval_start) * 1000),
        len(queries),
        len(search_items),
        len(candidates),
        len(selected),
    )

    return selected

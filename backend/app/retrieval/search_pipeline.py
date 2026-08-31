from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.evidence_builder import (
    build_evidence_from_snippet,
    build_evidence_from_url,
)
from backend.app.retrieval.hybrid import retrieve_hybrid
from backend.app.retrieval.reranker import rerank_documents
from backend.app.retrieval.tavily_provider import TavilySearchProvider
from backend.app.retrieval.fusion import fuse_rankings
from backend.app.retrieval.diversity import select_diverse_sources
import logging
logger = logging.getLogger(__name__)
def run_search(
    query: str,
    max_results: int,
):
    provider = TavilySearchProvider()

    return provider.search(
        query=query,
        max_results=max_results,
    )


def search_and_retrieve(
    query: str,
    search_results: int = 10,
    top_k: int = 5,
    alternate_query: str | None = None,
) -> list[RetrievedEvidence]:
    queries = [query]

    if alternate_query and alternate_query != query:
        queries.append(alternate_query)

    # Run search queries concurrently
    search_result_groups = []

    with ThreadPoolExecutor(
        max_workers=min(2, len(queries))
    ) as executor:
        futures = [
            executor.submit(
                run_search,
                search_query,
                search_results,
            )
            for search_query in queries
        ]

        for future in as_completed(futures):
            try:
                search_result_groups.append(
                    future.result()
                )
            except Exception as exc:
                logger.warning(
                    "Search failed: %s",
                    exc,
            )

    # Deduplicate URLs
    seen_urls = set()
    search_items = []

    for results in search_result_groups:
        for result in results:
            if not result.url:
                continue

            if result.url in seen_urls:
                continue

            seen_urls.add(result.url)
            search_items.append(result)

    if not search_items:
        return []

    # Fetch pages concurrently
    candidates = []

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                build_evidence_from_url,
                result.url,
                result.title,
            ): result
            for result in search_items
        }

        for future in as_completed(futures):
            try:
                candidates.append(
                    future.result()
                )
            except Exception as exc:
                result = futures[future]

                logger.warning(
                    "Failed to build evidence from %s: %s",
                    result.url,
                    exc,
            )

            if result.snippet.strip():
                candidates.append(
                    build_evidence_from_snippet(
                        url=result.url,
                        title=result.title,
                        snippet=result.snippet,
                    )
                )

    if not candidates:
        return []

    # Stage 1: hybrid BM25 + dense ranking
    hybrid_ranked = retrieve_hybrid(
        query=query,
        documents=candidates,
        top_k=len(candidates),
    )

    # Stage 2: rerank only the strongest hybrid candidates
    rerank_count = min(
        max(top_k * 2, 6),
        len(hybrid_ranked),
    )

    rerank_candidates = hybrid_ranked[
        :rerank_count
    ]

    reranked = rerank_documents(
        query=query,
        documents=rerank_candidates,
        top_k=len(rerank_candidates),
    )
    fused_ranked = fuse_rankings(
        hybrid_ranked=rerank_candidates,
        reranked=reranked,
    )

    # Stage 3: enforce source diversity
    return select_diverse_sources(
        documents=fused_ranked,
        top_k=top_k,
    )
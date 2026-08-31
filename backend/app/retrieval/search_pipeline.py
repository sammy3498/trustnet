from concurrent.futures import ThreadPoolExecutor, as_completed

from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.bm25 import retrieve_bm25
from backend.app.retrieval.evidence_builder import build_evidence_from_url
from backend.app.retrieval.tavily_provider import TavilySearchProvider


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

    # Run raw-claim search and generated-query search concurrently
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
            except Exception:
                continue

    # Deduplicate URLs across both searches
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

    # Fetch webpages concurrently
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
            except Exception:
                continue

    if not candidates:
        return []

    # Rank all fetched candidates
    ranked = retrieve_bm25(
        query=query,
        documents=candidates,
        top_k=len(candidates),
    )

    # Keep evidence from distinct source domains
    diverse_results = []
    seen_sources = set()

    for item in ranked:
        if item.source in seen_sources:
            continue

        seen_sources.add(item.source)
        diverse_results.append(item)

        if len(diverse_results) == top_k:
            break

    return diverse_results
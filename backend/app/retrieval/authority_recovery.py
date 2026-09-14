import logging
import re
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
    build_evidence_from_url,
)
from backend.app.retrieval.search_pipeline import (
    build_authority_search_query,
    is_search_result_usable,
    run_search,
)
from backend.app.retrieval.temporal import (
    is_publication_date_after_cutoff,
    is_text_explicitly_after_cutoff,
)


logger = logging.getLogger(
    __name__
)


AUTHORITY_RECOVERY_MAX_RESULTS = 8
AUTHORITY_RECOVERY_MAX_EVIDENCE = 3

AUTHORITY_SEARCH_TIMEOUT_SECONDS = 4.0
AUTHORITY_FETCH_TIMEOUT_SECONDS = 7.0

AUTHORITY_RECOVERY_DEADLINE_RESERVE_SECONDS = (
    2.0
)


RELIABLE_QUALITIES = {
    "high",
    "medium",
}


RELATION_QUERY_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "as",
    "that",
    "this",
    "these",
    "those",
}


def normalize_relation_term(
    term: str,
) -> str:
    value = term.lower()

    if (
        len(value) > 4
        and value.endswith(
            "ies"
        )
    ):
        return (
            value[:-3]
            + "y"
        )

    if (
        len(value) > 4
        and value.endswith(
            "s"
        )
        and not value.endswith(
            (
                "ss",
                "us",
                "is",
            )
        )
    ):
        return value[:-1]

    return value


def build_neutral_relation_query(
    claim: str,
) -> str | None:
    words = re.findall(
        r"[A-Za-z]+",
        claim,
    )

    terms = []

    for word in words:
        lower = word.lower()

        if (
            lower
            in RELATION_QUERY_STOPWORDS
        ):
            continue

        normalized = (
            normalize_relation_term(
                lower
            )
        )

        if (
            normalized
            not in terms
        ):
            terms.append(
                normalized
            )

    if len(
        terms
    ) < 2:
        return None

    relation = " ".join(
        terms[:8]
    )

    return (
        f"{relation} "
        "official authoritative source"
    )


def build_recovery_queries(
    claim: str,
) -> list[str]:
    queries = []

    neutral_query = (
        build_neutral_relation_query(
            claim
        )
    )

    if neutral_query:
        queries.append(
            neutral_query
        )

    proposition_query = (
        build_authority_search_query(
            claim
        )
    )

    if (
        proposition_query
        and proposition_query
        not in queries
    ):
        queries.append(
            proposition_query
        )

    return queries


def get_evidence_publisher(
    evidence: RetrievedEvidence,
) -> str:
    publisher = (
        get_publisher_domain(
            evidence.source
            or evidence.url
        )
    )

    if publisher:
        return publisher

    return evidence.url


def is_existing_reliable_webpage(
    evidence: RetrievedEvidence,
) -> bool:
    return (
        evidence.evidence_origin
        == "webpage"
        and evidence.quality
        in RELIABLE_QUALITIES
    )


def get_remaining_recovery_time(
    deadline: float | None,
) -> float | None:
    if deadline is None:
        return None

    remaining = (
        deadline
        - time.perf_counter()
        - AUTHORITY_RECOVERY_DEADLINE_RESERVE_SECONDS
    )

    return max(
        0.0,
        remaining,
    )


def resolve_stage_timeout(
    stage_cap: float,
    deadline: float | None,
) -> float:
    remaining = (
        get_remaining_recovery_time(
            deadline
        )
    )

    if remaining is None:
        return stage_cap

    return max(
        0.0,
        min(
            stage_cap,
            remaining,
        ),
    )


def run_authority_searches(
    queries: list[str],
    end_date: str | None,
    deadline: float | None,
):
    if not queries:
        return []

    timeout_seconds = (
        resolve_stage_timeout(
            AUTHORITY_SEARCH_TIMEOUT_SECONDS,
            deadline,
        )
    )

    if timeout_seconds < 0.25:
        return []

    executor = ThreadPoolExecutor(
        max_workers=min(
            2,
            len(
                queries
            ),
        )
    )

    futures = {
        executor.submit(
            run_search,
            query,
            AUTHORITY_RECOVERY_MAX_RESULTS,
            end_date,
        ): query
        for query in queries
    }

    results_by_query = {}

    try:
        try:
            for future in as_completed(
                futures,
                timeout=timeout_seconds,
            ):
                query = futures[
                    future
                ]

                try:
                    results_by_query[
                        query
                    ] = future.result()

                except Exception as exc:
                    logger.warning(
                        "Authority recovery search "
                        "failed for query '%s': %s",
                        query,
                        exc,
                    )

        except FuturesTimeoutError:
            logger.warning(
                "Authority recovery search "
                "stage exceeded %.2f seconds.",
                timeout_seconds,
            )

    finally:
        for future in futures:
            if not future.done():
                future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    merged = []

    seen_urls = set()

    for query in queries:
        for result in (
            results_by_query.get(
                query,
                [],
            )
        ):
            if result.url in seen_urls:
                continue

            seen_urls.add(
                result.url
            )

            merged.append(
                result
            )

    return merged


def select_authority_fetch_candidates(
    results,
    existing_evidence: list[
        RetrievedEvidence
    ],
    end_date: str | None,
):
    existing_urls = {
        item.url
        for item in existing_evidence
    }

    existing_publishers = {
        get_evidence_publisher(
            item
        )
        for item in existing_evidence
        if is_existing_reliable_webpage(
            item
        )
    }

    candidates = []

    candidate_publishers = set()

    for result in results:
        if (
            len(
                candidates
            )
            >= AUTHORITY_RECOVERY_MAX_EVIDENCE
        ):
            break

        if not is_search_result_usable(
            result=result,
            end_date=end_date,
        ):
            continue

        if result.url in existing_urls:
            continue

        quality = (
            classify_source_quality(
                result.url
            )
        )

        if (
            quality
            not in RELIABLE_QUALITIES
        ):
            continue

        publisher = (
            get_publisher_domain(
                result.url
            )
        )

        if not publisher:
            publisher = result.url

        if (
            publisher
            in existing_publishers
        ):
            continue

        if (
            publisher
            in candidate_publishers
        ):
            continue

        candidates.append(
            result
        )

        candidate_publishers.add(
            publisher
        )

    return candidates


def is_recovered_page_usable(
    evidence: RetrievedEvidence,
    end_date: str | None,
) -> bool:
    if (
        evidence.evidence_origin
        != "webpage"
    ):
        return False

    if (
        evidence.quality
        not in RELIABLE_QUALITIES
    ):
        return False

    if is_publication_date_after_cutoff(
        published_date=(
            evidence.published_date
        ),
        end_date=end_date,
    ):
        return False

    if (
        not evidence.published_date
        and is_text_explicitly_after_cutoff(
            text=evidence.text,
            end_date=end_date,
        )
    ):
        return False

    return True


def fetch_authority_pages(
    candidates,
    end_date: str | None,
    deadline: float | None,
) -> list[
    RetrievedEvidence
]:
    if not candidates:
        return []

    timeout_seconds = (
        resolve_stage_timeout(
            AUTHORITY_FETCH_TIMEOUT_SECONDS,
            deadline,
        )
    )

    if timeout_seconds < 0.5:
        return []

    executor = ThreadPoolExecutor(
        max_workers=min(
            len(
                candidates
            ),
            AUTHORITY_RECOVERY_MAX_EVIDENCE,
        )
    )

    futures = {
        executor.submit(
            build_evidence_from_url,
            result.url,
            result.title,
        ): result
        for result in candidates
    }

    recovered = []

    try:
        try:
            for future in as_completed(
                futures,
                timeout=timeout_seconds,
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
                        "Authority page fetch "
                        "failed for %s: %s",
                        result.url,
                        exc,
                    )

                    continue

                if not is_recovered_page_usable(
                    evidence=evidence,
                    end_date=end_date,
                ):
                    continue

                recovered.append(
                    evidence
                )

                if (
                    len(
                        recovered
                    )
                    >= AUTHORITY_RECOVERY_MAX_EVIDENCE
                ):
                    break

        except FuturesTimeoutError:
            logger.warning(
                "Authority page fetch "
                "stage exceeded %.2f seconds.",
                timeout_seconds,
            )

    finally:
        for future in futures:
            if not future.done():
                future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )

    return recovered


def recover_authority_evidence(
    claim: str,
    existing_evidence: list[
        RetrievedEvidence
    ],
    end_date: str | None = None,
    deadline: float | None = None,
) -> list[
    RetrievedEvidence
]:
    remaining = (
        get_remaining_recovery_time(
            deadline
        )
    )

    if (
        remaining is not None
        and remaining < 0.5
    ):
        return []

    queries = (
        build_recovery_queries(
            claim
        )
    )

    if not queries:
        return []

    results = (
        run_authority_searches(
            queries=queries,
            end_date=end_date,
            deadline=deadline,
        )
    )

    if not results:
        return []

    candidates = (
        select_authority_fetch_candidates(
            results=results,
            existing_evidence=(
                existing_evidence
            ),
            end_date=end_date,
        )
    )

    if not candidates:
        return []

    recovered = (
        fetch_authority_pages(
            candidates=candidates,
            end_date=end_date,
            deadline=deadline,
        )
    )

    if recovered:
        logger.warning(
            "Authority recovery added "
            "%s fetched independent "
            "reliable webpage source(s).",
            len(
                recovered
            ),
        )

    return recovered

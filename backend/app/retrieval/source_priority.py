"""Prioritize topical primary evidence *before* spending the fetch budget.

A domain is never proof that a passage supports a claim. Selection combines
publisher quality, the facet for which a result was found, and topical overlap.
"""
import re
from functools import lru_cache

from backend.app.retrieval.excerpt import get_relevant_excerpt
from backend.app.evidence.domain import get_publisher_domain
from backend.app.evidence.provenance import classify_source_quality

STOPWORDS = set("the and for with from this that have has was were but which what when where who why official original primary source data evidence report about than a an in on is are to of by at as it can any no one how does do".split())
QUALITY = {"high": 2.0, "medium": 1.0, "low": 0.0}


def tokens(value: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(value).lower()) if len(t) > 2 and t not in STOPWORDS}


def overlap(text: str, query: str) -> float:
    required = tokens(query)
    if not required:
        return 0.0
    return len(required.intersection(tokens(text))) / len(required)


def priority(result, query: str, search_rank: int = 10) -> float:
    quality = QUALITY.get(classify_source_quality(result.url), 0)
    title_fit = overlap(result.title or "", query)
    snippet_fit = overlap(result.snippet or "", query)
    fit = max(title_fit, snippet_fit * 0.85)
    # Do not reward an authoritative but unrelated page.
    return 4.0 * fit + (quality if fit >= 0.16 else 0) + 0.5 / max(1, search_rank)


def prioritize_for_fetch(search_items, url_query_ranks, queries, max_items: int):
    """Reserve candidates across the queries, then fill with best available.

    Round-robin facet coverage prevents the first query's results from using
    the entire fetch budget.  Each group is ranked for its own query.
    """
    if max_items <= 0:
        return []
    indexed = {item.url: item for item in search_items}
    selected, seen, publishers = [], set(), set()

    def pick(item, require_new_publisher=False):
        if item.url in seen or len(selected) >= max_items:
            return False
        publisher = get_publisher_domain(item.url)
        if require_new_publisher and publisher and publisher in publishers:
            return False
        selected.append(item)
        seen.add(item.url)
        if publisher:
            publishers.add(publisher)
        return True

    # Reserve the best *relevant* candidate for every search facet first.
    for query in queries:
        matches = [(priority(item, query, url_query_ranks.get(item.url, {}).get(query, 100)), item)
                   for item in search_items if query in url_query_ranks.get(item.url, {})]
        matches.sort(key=lambda p: p[0], reverse=True)
        for _, item in matches:
            if pick(item, require_new_publisher=True):
                break
        else:
            for _, item in matches:
                if pick(item):
                    break
    # Score all results against the facet(s) that actually returned them.
    remaining = sorted(
        indexed.values(),
        key=lambda item: max((priority(item, q, rank) for q, rank in url_query_ranks.get(item.url, {}).items()), default=0),
        reverse=True,
    )
    for require_unique in (True, False):
        for item in remaining:
            pick(item, require_new_publisher=require_unique)
    return selected[:max_items]


def select_facet_evidence(ranked, fallback, query_ranks, queries, top_k):
    """Choose independently relevant documents for each verification facet."""
    if top_k <= 0:
        return []
    documents = list(dict((d.url, d) for d in (ranked + fallback)).values())
    positions = {d.url: i for i, d in enumerate(ranked + fallback)}
    chosen, urls, publishers = [], set(), set()

    by_url = {doc.url: doc for doc in documents}

    @lru_cache(maxsize=256)
    def fit(url, q):
        doc = by_url[url]
        try:
            # A matching passage may be far below the first 1,300 characters
            # of a primary-source report or statistics table.
            excerpt = get_relevant_excerpt(text=doc.text, query=q, max_chars=1000)
        except Exception:
            excerpt = (doc.text or "")[:1300]
        return max(overlap(doc.title or "", q), overlap(excerpt, q) * 0.92)

    def score(doc, q):
        src = query_ranks.get(doc.url, {})
        rank = src.get(q, 100)
        relevance = fit(doc.url, q)
        authority = QUALITY.get(doc.quality, 0)
        # Direct page content is preferable to a truncated snippet.
        origin = 0.45 if doc.evidence_origin == "webpage" else 0
        return 5.0 * relevance + authority * (1 if relevance >= 0.15 else 0) + origin - 0.012 * positions.get(doc.url, 100) - 0.025 * rank

    def add(doc, unique=True):
        pub = get_publisher_domain(doc.source or doc.url)
        if doc.url in urls or len(chosen) >= top_k or (unique and pub and pub in publishers):
            return False
        chosen.append(doc)
        urls.add(doc.url)
        if pub:
            publishers.add(pub)
        return True

    for query in queries:
        matching = [d for d in documents if query in query_ranks.get(d.url, {})]
        if not matching:
            continue
        matching.sort(key=lambda d: score(d, query), reverse=True)
        # Per-facet reservation may use same publisher twice only as a fallback.
        for doc in matching:
            if add(doc):
                break
        else:
            for doc in matching:
                if add(doc, unique=False):
                    break
    remaining = sorted(documents, key=lambda d: max((score(d, q) for q in queries), default=0), reverse=True)
    for unique in (True, False):
        for doc in remaining:
            add(doc, unique=unique)
    for doc in chosen:
        candidates = query_ranks.get(doc.url, {})
        if candidates:
            doc.matched_query = max(candidates, key=lambda q: score(doc, q))
    return chosen[:top_k]

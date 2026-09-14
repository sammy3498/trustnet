from types import SimpleNamespace

from backend.app.evidence.domain import (
    get_publisher_domain,
)
from backend.app.retrieval.search_pipeline import (
    add_authority_search_query,
    select_fetch_candidates,
)


def search_result(
    url: str,
):
    return SimpleNamespace(
        url=url,
        title="Test result",
        snippet="Test snippet",
    )


def test_authority_query_is_added_when_slot_available():
    queries = [
        "main query",
        "verification query",
    ]

    result = (
        add_authority_search_query(
            queries=queries,
            query="Earth orbits the Sun",
        )
    )

    assert result == [
        "main query",
        "verification query",
        (
            "Earth orbits the Sun "
            "official primary source"
        ),
    ]


def test_authority_query_does_not_exceed_query_limit():
    queries = [
        "one",
        "two",
        "three",
        "four",
    ]

    result = (
        add_authority_search_query(
            queries=queries,
            query="main",
        )
    )

    assert result == queries


def test_fetch_selection_prefers_reliable_independent_sources():
    low = search_result(
        "https://example.com/claim"
    )

    nasa = search_result(
        "https://science.nasa.gov/a"
    )

    who = search_result(
        "https://who.int/b"
    )

    nasa_second = (
        search_result(
            "https://www.nasa.gov/c"
        )
    )

    items = [
        low,
        nasa,
        who,
        nasa_second,
    ]

    ranks = {
        low.url: {
            "query": 1,
        },
        nasa.url: {
            "query": 2,
        },
        who.url: {
            "query": 3,
        },
        nasa_second.url: {
            "query": 1,
        },
    }

    selected = (
        select_fetch_candidates(
            search_items=items,
            url_query_ranks=ranks,
            max_candidates=3,
        )
    )

    urls = [
        item.url
        for item in selected
    ]

    publishers = [
        get_publisher_domain(
            item.url
        )
        for item in selected
    ]

    assert who.url in urls

    assert (
        publishers.count(
            "nasa.gov"
        )
        == 1
    )

    assert (
        "who.int"
        in publishers
    )

    assert (
        "example.com"
        in publishers
    )

    assert (
        nasa_second.url
        in urls
    )

    assert (
        nasa.url
        not in urls
    )


def test_strong_search_relevance_is_not_ignored():
    direct_low = search_result(
        "https://example.com/direct"
    )

    weaker_high = search_result(
        "https://nasa.gov/background"
    )

    ranks = {
        direct_low.url: {
            "query": 1,
        },
        weaker_high.url: {
            "query": 8,
        },
    }

    selected = (
        select_fetch_candidates(
            search_items=[
                direct_low,
                weaker_high,
            ],
            url_query_ranks=ranks,
            max_candidates=2,
        )
    )

    urls = [
        item.url
        for item in selected
    ]

    assert direct_low.url in urls

    assert weaker_high.url in urls
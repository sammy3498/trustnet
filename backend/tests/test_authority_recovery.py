import time
from types import SimpleNamespace

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval import (
    authority_recovery,
)


def existing_evidence(
    source: str,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title="Existing",
        url=(
            f"https://{source}/existing"
        ),
        source=source,
        text="Existing evidence.",
        quality="medium",
    )


def search_result(
    url: str,
    title: str = "Authority result",
):
    return SimpleNamespace(
        url=url,
        title=title,
        snippet=(
            "The Earth follows an orbit "
            "around the Sun."
        ),
    )


def fetched_evidence(
    result,
    quality: str = "medium",
):
    source = (
        result.url
        .split(
            "//",
            1,
        )[1]
        .split(
            "/",
            1,
        )[0]
    )

    return RetrievedEvidence(
        title=result.title,
        url=result.url,
        source=source,
        text=(
            "The Earth follows an orbit "
            "around the Sun."
        ),
        quality=quality,
        evidence_origin="webpage",
    )


def test_neutral_relation_query():
    query = (
        authority_recovery
        .build_neutral_relation_query(
            "The Sun orbits the Earth."
        )
    )

    assert query == (
        "sun orbit earth "
        "official authoritative source"
    )


def test_recovery_queries_put_neutral_query_first():
    queries = (
        authority_recovery
        .build_recovery_queries(
            "The Sun orbits the Earth."
        )
    )

    assert queries[
        0
    ] == (
        "sun orbit earth "
        "official authoritative source"
    )

    assert queries[
        1
    ] == (
        "The Sun orbits the Earth. "
        "official primary source"
    )


def test_neutral_query_can_recover_new_publisher(
    monkeypatch,
):
    calls = []

    def fake_search(
        query,
        max_results,
        end_date=None,
    ):
        calls.append(
            query
        )

        if query.startswith(
            "sun orbit earth"
        ):
            return [
                search_result(
                    (
                        "https://"
                        "courses.ems.psu.edu/"
                        "meteo300/node/681"
                    ),
                    "Earth Rotating Around the Sun",
                ),
            ]

        return [
            search_result(
                "https://osu.edu/repeated"
            )
        ]

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        fake_search,
    )

    monkeypatch.setattr(
        authority_recovery,
        "classify_source_quality",
        lambda url: "medium",
    )

    monkeypatch.setattr(
        authority_recovery,
        "build_evidence_from_url",
        lambda url, title: (
            fetched_evidence(
                SimpleNamespace(
                    url=url,
                    title=title,
                )
            )
        ),
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Sun orbits "
                "the Earth."
            ),
            existing_evidence=[
                existing_evidence(
                    "osu.edu"
                )
            ],
        )
    )

    assert len(
        recovered
    ) == 1

    assert (
        recovered[
            0
        ].source
        == "courses.ems.psu.edu"
    )

    assert any(
        query.startswith(
            "sun orbit earth"
        )
        for query in calls
    )


def test_recovery_fetches_new_reliable_page(
    monkeypatch,
):
    neutral_results = [
        search_result(
            "https://psu.edu/independent"
        ),
        search_result(
            "https://weak.example/test"
        ),
    ]

    proposition_results = [
        search_result(
            "https://osu.edu/repeated"
        ),
    ]

    def fake_search(
        query,
        max_results,
        end_date=None,
    ):
        if (
            "official authoritative"
            in query
        ):
            return neutral_results

        return proposition_results

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        fake_search,
    )

    def fake_quality(
        url,
    ):
        if (
            "weak.example"
            in url
        ):
            return "low"

        return "medium"

    monkeypatch.setattr(
        authority_recovery,
        "classify_source_quality",
        fake_quality,
    )

    fetch_calls = []

    def fake_fetch(
        url,
        title,
    ):
        fetch_calls.append(
            url
        )

        return fetched_evidence(
            SimpleNamespace(
                url=url,
                title=title,
            )
        )

    monkeypatch.setattr(
        authority_recovery,
        "build_evidence_from_url",
        fake_fetch,
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            existing_evidence=[
                existing_evidence(
                    "osu.edu"
                )
            ],
        )
    )

    assert len(
        recovered
    ) == 1

    assert (
        recovered[
            0
        ].source
        == "psu.edu"
    )

    assert fetch_calls == [
        "https://psu.edu/independent"
    ]


def test_low_quality_search_result_is_not_fetched(
    monkeypatch,
):
    results = [
        search_result(
            "https://weak.example/test"
        )
    ]

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        lambda query, max_results, end_date=None: (
            results
        ),
    )

    monkeypatch.setattr(
        authority_recovery,
        "classify_source_quality",
        lambda url: "low",
    )

    def should_not_fetch(
        url,
        title,
    ):
        raise AssertionError(
            "Low-quality search result "
            "must not be fetched."
        )

    monkeypatch.setattr(
        authority_recovery,
        "build_evidence_from_url",
        should_not_fetch,
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            existing_evidence=[],
        )
    )

    assert recovered == []


def test_failed_page_fetch_is_not_replaced_by_snippet(
    monkeypatch,
):
    results = [
        search_result(
            "https://psu.edu/test"
        )
    ]

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        lambda query, max_results, end_date=None: (
            results
        ),
    )

    monkeypatch.setattr(
        authority_recovery,
        "classify_source_quality",
        lambda url: "medium",
    )

    def failing_fetch(
        url,
        title,
    ):
        raise RuntimeError(
            "Fetch failed."
        )

    monkeypatch.setattr(
        authority_recovery,
        "build_evidence_from_url",
        failing_fetch,
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            existing_evidence=[],
        )
    )

    assert recovered == []


def test_fetched_low_quality_page_is_rejected(
    monkeypatch,
):
    results = [
        search_result(
            "https://psu.edu/test"
        )
    ]

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        lambda query, max_results, end_date=None: (
            results
        ),
    )

    monkeypatch.setattr(
        authority_recovery,
        "classify_source_quality",
        lambda url: "medium",
    )

    monkeypatch.setattr(
        authority_recovery,
        "build_evidence_from_url",
        lambda url, title: RetrievedEvidence(
            title=title,
            url=url,
            source="psu.edu",
            text="Test evidence.",
            quality="low",
            evidence_origin="webpage",
        ),
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            existing_evidence=[],
        )
    )

    assert recovered == []


def test_recovery_skips_when_deadline_is_spent(
    monkeypatch,
):
    def should_not_run(
        query,
        max_results,
        end_date=None,
    ):
        raise AssertionError(
            "Search should not run."
        )

    monkeypatch.setattr(
        authority_recovery,
        "run_search",
        should_not_run,
    )

    recovered = (
        authority_recovery
        .recover_authority_evidence(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            existing_evidence=[],
            deadline=(
                time.perf_counter()
                - 1.0
            ),
        )
    )

    assert recovered == []
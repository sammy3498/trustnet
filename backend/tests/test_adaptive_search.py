from types import SimpleNamespace

from backend.app.retrieval import (
    search_pipeline,
)


def search_result(
    url: str,
):
    return SimpleNamespace(
        url=url,
        title="Test result",
        snippet="Test snippet",
    )


def test_primary_search_stops_after_one_call(
    monkeypatch,
):
    calls = []

    def fake_run_search(
        query,
        max_results,
        end_date=None,
    ):
        calls.append(
            query
        )

        return [
            search_result(
                "https://www.nasa.gov/a"
            ),
            search_result(
                "https://www.who.int/b"
            ),
        ]

    monkeypatch.setattr(
        search_pipeline,
        "run_search",
        fake_run_search,
    )

    (
        results,
        executed,
    ) = (
        search_pipeline
        .run_adaptive_searches(
            query=(
                "Earth orbits the Sun"
            ),
            candidate_queries=[
                "Earth orbits the Sun"
            ],
            search_results=5,
        )
    )

    assert len(
        calls
    ) == 1

    assert executed == [
        "Earth orbits the Sun"
    ]

    assert (
        "Earth orbits the Sun"
        in results
    )


def test_weak_primary_triggers_authority_search(
    monkeypatch,
):
    calls = []

    def fake_run_search(
        query,
        max_results,
        end_date=None,
    ):
        calls.append(
            query
        )

        if len(
            calls
        ) == 1:
            return [
                search_result(
                    (
                        "https://"
                        "example.com/a"
                    )
                )
            ]

        return [
            search_result(
                "https://www.nasa.gov/a"
            ),
            search_result(
                "https://www.who.int/b"
            ),
        ]

    monkeypatch.setattr(
        search_pipeline,
        "run_search",
        fake_run_search,
    )

    (
        _,
        executed,
    ) = (
        search_pipeline
        .run_adaptive_searches(
            query=(
                "Earth orbits the Sun"
            ),
            candidate_queries=[
                "Earth orbits the Sun"
            ],
            search_results=5,
        )
    )

    assert len(
        calls
    ) == 2

    assert executed == [
        "Earth orbits the Sun",
        (
            "Earth orbits the Sun "
            "official primary source"
        ),
    ]


def test_same_publisher_does_not_count_as_independent(
    monkeypatch,
):
    calls = []

    def fake_run_search(
        query,
        max_results,
        end_date=None,
    ):
        calls.append(
            query
        )

        if len(
            calls
        ) == 1:
            return [
                search_result(
                    (
                        "https://"
                        "science.nasa.gov/a"
                    )
                ),
                search_result(
                    (
                        "https://"
                        "www.nasa.gov/b"
                    )
                ),
            ]

        return [
            search_result(
                "https://www.who.int/c"
            )
        ]

    monkeypatch.setattr(
        search_pipeline,
        "run_search",
        fake_run_search,
    )

    (
        _,
        executed,
    ) = (
        search_pipeline
        .run_adaptive_searches(
            query="test claim",
            candidate_queries=[
                "test claim"
            ],
            search_results=5,
        )
    )

    assert len(
        calls
    ) == 2

    assert len(
        executed
    ) == 2


def test_run_search_has_no_hidden_retry(
    monkeypatch,
):
    calls = []

    class FakeProvider:
        def search(
            self,
            query,
            max_results,
            end_date=None,
        ):
            calls.append(
                {
                    "query": query,
                    "max_results": (
                        max_results
                    ),
                    "end_date": end_date,
                }
            )

            return []

    monkeypatch.setattr(
        search_pipeline,
        "TavilySearchProvider",
        FakeProvider,
    )

    result = (
        search_pipeline.run_search(
            query="test",
            max_results=5,
            end_date="2025-01-01",
        )
    )

    assert result == []

    assert len(
        calls
    ) == 1

    assert (
        calls[0][
            "end_date"
        ]
        == "2025-01-01"
    )
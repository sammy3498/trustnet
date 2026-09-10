from backend.app.retrieval.search_pipeline import (
    build_search_queries,
    is_excluded_search_result,
)


def test_verification_queries_are_added():
    result = build_search_queries(
        query="main",
        alternate_query="alternate",
        verification_queries=[
            "verify one",
            "verify two",
        ],
    )

    assert result == [
        "alternate",
        "verify one",
        "verify two",
        "main",
    ]


def test_duplicate_queries_are_removed():
    result = build_search_queries(
        query="main",
        alternate_query="main",
        verification_queries=[
            "main",
            "verify",
        ],
    )

    assert result == [
        "main",
        "verify",
    ]


def test_query_limit_prioritizes_verification_queries():
    result = build_search_queries(
        query="main",
        alternate_query="alternate",
        verification_queries=[
            "verify one",
            "verify two",
            "verify three",
        ],
    )

    assert result == [
        "alternate",
        "verify one",
        "verify two",
        "verify three",
    ]


def test_query_limit_is_four():
    result = build_search_queries(
        query="main",
        alternate_query="alternate",
        verification_queries=[
            "q1",
            "q2",
            "q3",
            "q4",
        ],
    )

    assert len(result) == 4


def test_exclusion_disabled_by_default():
    assert not (
        is_excluded_search_result(
            url=(
                "https://huggingface.co/"
                "datasets/example"
            ),
            title="Example dataset",
        )
    )


def test_exclusion_matches_url():
    assert (
        is_excluded_search_result(
            url=(
                "https://huggingface.co/"
                "datasets/averitec"
            ),
            title="Dataset",
            excluded_patterns=[
                "averitec",
            ],
        )
    )


def test_exclusion_matches_title():
    assert (
        is_excluded_search_result(
            url=(
                "https://example.com/page"
            ),
            title=(
                "AVeriTeC benchmark data"
            ),
            excluded_patterns=[
                "averitec",
            ],
        )
    )


def test_exclusion_is_case_insensitive():
    assert (
        is_excluded_search_result(
            url=(
                "https://example.com/"
                "AVERITEC/data"
            ),
            title="Example",
            excluded_patterns=[
                "averitec",
            ],
        )
    )
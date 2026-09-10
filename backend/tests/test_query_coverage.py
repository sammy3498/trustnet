from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.retrieval import (
    query_coverage,
)
from backend.app.retrieval.query_coverage import (
    select_query_aware_evidence,
)


def make_document(
    name: str,
    source: str,
    quality: str = "medium",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=name,
        url=f"https://{source}/{name}",
        source=source,
        text=f"Evidence for {name}",
        quality=quality,
    )


def test_verification_query_gets_coverage(
    monkeypatch,
):
    global_document = make_document(
        "global",
        "global.com",
    )

    verification_document = (
        make_document(
            "verification",
            "verify.com",
        )
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                document,
                5.0,
            )
            for document in documents
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                global_document
            ],
            fallback_ranked=[
                global_document,
                verification_document,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                verification_document.url: {
                    "verification query": 1
                }
            },
            top_k=2,
        )
    )

    assert (
        verification_document
        in result
    )


def test_semantic_relevance_beats_search_rank(
    monkeypatch,
):
    weak = make_document(
        "weak",
        "weak.com",
    )

    strong = make_document(
        "strong",
        "strong.com",
    )

    def fake_scores(
        query,
        documents,
    ):
        return [
            (
                strong,
                6.0,
            ),
            (
                weak,
                -2.0,
            ),
        ]

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        fake_scores,
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                weak,
                strong,
            ],
            fallback_ranked=[
                weak,
                strong,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                weak.url: {
                    "verification query": 1
                },
                strong.url: {
                    "verification query": 5
                },
            },
            top_k=1,
        )
    )

    assert result == [
        strong
    ]


def test_irrelevant_low_quality_candidate_is_not_forced(
    monkeypatch,
):
    global_document = make_document(
        "global",
        "global.com",
    )

    irrelevant = make_document(
        "irrelevant",
        "irrelevant.com",
        quality="low",
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                irrelevant,
                -4.0,
            )
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                global_document
            ],
            fallback_ranked=[
                irrelevant,
                global_document,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                irrelevant.url: {
                    "verification query": 1
                }
            },
            top_k=1,
        )
    )

    assert result == [
        global_document
    ]


def test_reliable_top_candidate_can_recover_negative_score(
    monkeypatch,
):
    global_document = make_document(
        "global",
        "global.com",
    )

    reliable = make_document(
        "reliable",
        "reliable.com",
        quality="medium",
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                reliable,
                -0.8,
            )
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                global_document
            ],
            fallback_ranked=[
                reliable,
                global_document,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                reliable.url: {
                    "verification query": 1
                }
            },
            top_k=1,
        )
    )

    assert result == [
        reliable
    ]


def test_reliable_candidate_not_recovered_when_semantically_weaker(
    monkeypatch,
):
    global_document = make_document(
        "global",
        "global.com",
    )

    reliable = make_document(
        "reliable",
        "reliable.com",
        quality="medium",
    )

    stronger_low_quality = (
        make_document(
            "stronger",
            "stronger.com",
            quality="low",
        )
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                reliable,
                -4.0,
            ),
            (
                stronger_low_quality,
                -1.0,
            ),
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                global_document
            ],
            fallback_ranked=[
                reliable,
                stronger_low_quality,
                global_document,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                reliable.url: {
                    "verification query": 1
                },
                stronger_low_quality.url: {
                    "verification query": 2
                },
            },
            top_k=1,
        )
    )

    assert result == [
        global_document
    ]


def test_reliable_negative_rank_three_is_not_recovered(
    monkeypatch,
):
    global_document = make_document(
        "global",
        "global.com",
    )

    reliable = make_document(
        "reliable",
        "reliable.com",
        quality="high",
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                reliable,
                -0.5,
            )
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                global_document
            ],
            fallback_ranked=[
                reliable,
                global_document,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                reliable.url: {
                    "verification query": 3
                }
            },
            top_k=1,
        )
    )

    assert result == [
        global_document
    ]


def test_search_rank_breaks_equal_score_tie(
    monkeypatch,
):
    first = make_document(
        "first",
        "first.com",
    )

    second = make_document(
        "second",
        "second.com",
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                second,
                3.0,
            ),
            (
                first,
                3.0,
            ),
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                first,
                second,
            ],
            fallback_ranked=[
                first,
                second,
            ],
            verification_queries=[
                "verification query"
            ],
            url_query_ranks={
                first.url: {
                    "verification query": 1
                },
                second.url: {
                    "verification query": 3
                },
            },
            top_k=1,
        )
    )

    assert result == [
        first
    ]


def test_top_k_is_respected(
    monkeypatch,
):
    first = make_document(
        "first",
        "first.com",
    )

    second = make_document(
        "second",
        "second.com",
    )

    third = make_document(
        "third",
        "third.com",
    )

    monkeypatch.setattr(
        query_coverage,
        "score_documents_for_query",
        lambda query, documents: [
            (
                document,
                5.0,
            )
            for document in documents
        ],
    )

    result = (
        select_query_aware_evidence(
            primary_ranked=[
                first,
                second,
                third,
            ],
            fallback_ranked=[
                first,
                second,
                third,
            ],
            verification_queries=[],
            url_query_ranks={},
            top_k=2,
        )
    )

    assert len(
        result
    ) == 2
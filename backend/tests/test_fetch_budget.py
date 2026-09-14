import time
from types import SimpleNamespace

from backend.app.evidence.models import (
    RetrievedEvidence,
)
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


def test_fetch_stage_harvests_fast_page_before_slow_page(
    monkeypatch,
):
    slow = search_result(
        "https://slow.example/a"
    )

    fast = search_result(
        "https://fast.example/b"
    )

    def fake_build_evidence(
        url,
        title,
    ):
        if "slow.example" in url:
            time.sleep(
                0.15
            )

        else:
            time.sleep(
                0.01
            )

        return RetrievedEvidence(
            title=title,
            url=url,
            source=url,
            text="Evidence text",
            quality="low",
            evidence_origin=(
                "webpage"
            ),
        )

    monkeypatch.setattr(
        search_pipeline,
        "build_evidence_from_url",
        fake_build_evidence,
    )

    result = (
        search_pipeline
        .fetch_evidence_pages(
            fetch_items=[
                slow,
                fast,
            ],
            timeout_seconds=0.05,
        )
    )

    assert fast.url in result

    assert slow.url not in result


def test_default_fetch_candidate_budget_is_five():
    items = [
        search_result(
            f"https://source{i}.com/a"
        )
        for i in range(
            6
        )
    ]

    ranks = {
        item.url: {
            "query": (
                index + 1
            )
        }
        for index, item
        in enumerate(
            items
        )
    }

    selected = (
        search_pipeline
        .select_fetch_candidates(
            search_items=items,
            url_query_ranks=ranks,
        )
    )

    assert len(
        selected
    ) == 5


def test_normal_fetch_budget_is_bounded():
    assert (
        search_pipeline
        .FETCH_STAGE_TIMEOUT_SECONDS
        <= 3.5
    )
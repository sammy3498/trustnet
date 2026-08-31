from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.diversity import select_diverse_sources


def make_item(source: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=source,
        url=f"https://{source}/test",
        source=source,
        text="test evidence",
        quality="high",
    )


def test_same_publisher_is_deduplicated():
    documents = [
        make_item("science.nasa.gov"),
        make_item("www.nasa.gov"),
        make_item("reuters.com"),
    ]

    result = select_diverse_sources(
        documents,
        top_k=3,
    )

    assert len(result) == 2
    assert result[0].source == "science.nasa.gov"
    assert result[1].source == "reuters.com"


def test_top_k_is_respected():
    documents = [
        make_item("nasa.gov"),
        make_item("reuters.com"),
        make_item("bbc.com"),
    ]

    result = select_diverse_sources(
        documents,
        top_k=2,
    )

    assert len(result) == 2
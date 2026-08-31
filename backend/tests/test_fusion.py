from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.fusion import fuse_rankings


def make_item(name: str) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=name,
        url=f"https://{name}.com",
        source=f"{name}.com",
        text="test",
        quality="low",
    )


def test_fusion_preserves_strong_hybrid_candidate():
    a = make_item("a")
    b = make_item("b")
    c = make_item("c")

    hybrid_ranked = [
        a,
        b,
        c,
    ]

    reranked = [
        c,
        b,
        a,
    ]

    result = fuse_rankings(
        hybrid_ranked=hybrid_ranked,
        reranked=reranked,
    )

    assert result[0] == a


def test_fusion_returns_all_documents():
    a = make_item("a")
    b = make_item("b")
    c = make_item("c")

    result = fuse_rankings(
        hybrid_ranked=[a, b, c],
        reranked=[c, b, a],
    )

    assert len(result) == 3
    assert set(item.url for item in result) == {
        a.url,
        b.url,
        c.url,
    }
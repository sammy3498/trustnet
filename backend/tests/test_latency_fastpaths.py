"""Offline regression tests for the Normal/Deep latency improvements."""

from backend.app.analysis import pipeline as analysis
from backend.app.api.routes import build_public_evidence
from backend.app.evidence.models import RetrievedEvidence
from backend.app.utils.config import get_mode_limits


def make_evidence(n: int) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=f"Source {n}",
        url=f"https://example{n}.org/report",
        source=f"example{n}.org",
        text="A factual source about this claim.",
        quality="high",
    )


def fake_completed_result(**kwargs) -> dict:
    return {
        "claim": kwargs["claim"],
        "overall_score": 91,
        "label": "Highly Credible",
        "factors": None,
        "flags": [],
        "uncertainty": {"gate_triggered": False, "level": "low", "reason": "OK"},
        "evidence_analysis": {"evidence_assessments": []},
        "llm_timings": {},
    }


def test_deep_direct_claim_avoids_preparation_and_extra_search(monkeypatch):
    monkeypatch.delenv("TRUSTNET_DEEP_LLM_PREPARATION", raising=False)
    monkeypatch.setattr(
        analysis, "prepare_claim",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unneeded LLM prep")),
    )
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [make_evidence(1), make_evidence(2), make_evidence(3)]

    monkeypatch.setattr(analysis, "search_and_retrieve", retrieve)
    monkeypatch.setattr(analysis, "analyze_claim_with_llm", fake_completed_result)
    result = analysis.analyze_claim("The Earth orbits the Sun.", mode="deep")
    assert len(calls) == 1
    assert len(result["evidence"]) == 3
    assert result["mode"] == "deep"
    assert result["claim_analysis"]["is_clear"] is True
    assert result["latency_ms"] >= 0


def test_normal_avoids_preparation_and_extra_search(monkeypatch):
    monkeypatch.setattr(
        analysis, "prepare_claim",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Unneeded LLM prep")),
    )
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [make_evidence(1), make_evidence(2)]

    monkeypatch.setattr(analysis, "search_and_retrieve", retrieve)
    monkeypatch.setattr(analysis, "analyze_claim_with_llm", fake_completed_result)
    result = analysis.analyze_claim("The Earth orbits the Sun.", mode="normal")
    assert len(calls) == 1
    assert result["mode"] == "normal"


def test_deep_tries_one_followup_if_initial_retrieval_empty(monkeypatch):
    monkeypatch.delenv("TRUSTNET_DEEP_LLM_PREPARATION", raising=False)
    calls = []

    def retrieve(**kwargs):
        calls.append(kwargs)
        return [] if len(calls) == 1 else [make_evidence(1), make_evidence(2)]

    monkeypatch.setattr(analysis, "search_and_retrieve", retrieve)
    monkeypatch.setattr(analysis, "analyze_claim_with_llm", fake_completed_result)
    result = analysis.analyze_claim("A claim requiring a second query", mode="deep")
    assert len(calls) == 2
    assert len(result["evidence"]) == 2


def test_public_api_retains_unassessed_evidence_without_claiming_stance():
    result = {
        "evidence": [make_evidence(1).model_dump(), make_evidence(2).model_dump()],
        "evidence_analysis": {"evidence_assessments": []},
    }
    public = build_public_evidence(result)
    assert len(public) == 2
    assert all(item.stance == "neutral" for item in public)
    assert all("did not finish" in item.summary for item in public)


def test_public_api_keeps_assessed_stance_and_displays_remaining_unassessed():
    first, second = make_evidence(1), make_evidence(2)
    result = {
        "evidence": [first.model_dump(), second.model_dump()],
        "evidence_analysis": {
            "evidence_assessments": [
                {
                    "url": first.url,
                    "stance": "contradicts",
                    "reasoning": "The source contradicts the claim.",
                }
            ]
        },
    }
    public = build_public_evidence(result)
    assert len(public) == 2
    assert public[0].stance == "contradicts"
    assert public[1].stance == "neutral"


def test_mode_limits_leave_time_for_final_analysis():
    normal = get_mode_limits("normal")
    deep = get_mode_limits("deep")
    assert normal["search_stage_max_seconds"] <= 5
    assert normal["fetch_stage_max_seconds"] <= 4
    assert deep["search_stage_max_seconds"] + deep["fetch_stage_max_seconds"] <= 12
    assert deep["llm_call_max_seconds"] >= 30


def test_normal_retrieval_limits_fetches_and_skips_cross_encoder(monkeypatch):
    from backend.app.retrieval import search_pipeline as retrieval
    from backend.app.retrieval.search_provider import SearchResult

    search_results = [
        SearchResult(
            title=f"Source {n}",
            url=f"https://example{n}.org/report",
            snippet=f"Snippet confirming details from source {n}.",
        )
        for n in range(12)
    ]
    fetches = []
    snippets = []

    def get_page(url, title):
        fetches.append(url)
        return RetrievedEvidence(
            title=title, url=url, source=url.split('/')[2],
            text="Retrieved full page.", quality="high",
        )

    def from_snippet(url, title, snippet):
        snippets.append(url)
        return RetrievedEvidence(
            title=title, url=url, source=url.split('/')[2],
            text=snippet, quality="medium", evidence_origin="snippet",
        )

    def forbidden_rerank(*args, **kwargs):
        raise AssertionError("Normal Mode should not call the cross-encoder")

    monkeypatch.delenv("TRUSTNET_RERANK_NORMAL", raising=False)
    monkeypatch.setattr(retrieval, "run_search", lambda *args, **kwargs: search_results)
    monkeypatch.setattr(retrieval, "build_evidence_from_url", get_page)
    monkeypatch.setattr(retrieval, "build_evidence_from_snippet", from_snippet)
    monkeypatch.setattr(retrieval, "get_relevant_excerpt", lambda text, query, max_chars: text[:max_chars])
    monkeypatch.setattr(retrieval, "retrieve_hybrid", lambda query, documents, **kwargs: documents)
    monkeypatch.setattr(retrieval, "rerank_documents", forbidden_rerank)
    monkeypatch.setattr(retrieval, "select_query_aware_evidence", forbidden_rerank)

    evidence = retrieval.search_and_retrieve(
        query="A factual claim", search_results=12, top_k=3,
        verification_queries=["A factual claim"], mode="normal",
    )
    assert len(fetches) == 6
    assert len(snippets) == 6
    assert len(evidence) == 3


def test_deep_single_query_does_not_repeat_query_coverage_reranking(monkeypatch):
    from backend.app.retrieval import search_pipeline as retrieval
    from backend.app.retrieval.search_provider import SearchResult

    results = [
        SearchResult(title=f"Source {n}", url=f"https://example{n}.org/x", snippet="Source text")
        for n in range(15)
    ]
    rerank_calls = []

    monkeypatch.setattr(retrieval, "run_search", lambda *args, **kwargs: results)
    monkeypatch.setattr(
        retrieval, "build_evidence_from_url",
        lambda url, title: RetrievedEvidence(
            title=title, url=url, source=url.split('/')[2],
            text="Some source text", quality="medium"
        ),
    )
    monkeypatch.setattr(
        retrieval, "build_evidence_from_snippet",
        lambda url, title, snippet: RetrievedEvidence(
            title=title, url=url, source=url.split('/')[2],
            text=snippet, quality="medium", evidence_origin="snippet"
        ),
    )
    monkeypatch.setattr(retrieval, "get_relevant_excerpt", lambda text, query, max_chars: text)
    monkeypatch.setattr(retrieval, "retrieve_hybrid", lambda query, documents, **kwargs: documents)

    def rerank(query, documents, **kwargs):
        rerank_calls.append(query)
        return documents

    monkeypatch.setattr(retrieval, "rerank_documents", rerank)
    monkeypatch.setattr(
        retrieval, "select_query_aware_evidence",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Redundant query-coverage rerank")),
    )
    evidence = retrieval.search_and_retrieve(
        query="A factual claim", search_results=15, top_k=5,
        verification_queries=["A factual claim"], mode="deep",
    )
    assert len(rerank_calls) == 1
    assert len(evidence) == 5

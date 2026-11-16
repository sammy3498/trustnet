"""Structural regressions from screenshots; fixtures are not factual benchmarks."""
from backend.app.retrieval.verification_plan import plan_verification_queries, atomic_search_facets
from backend.app.retrieval.coverage_recovery import recovery_query
from backend.app.retrieval.evidence_builder import is_access_wall
from backend.app.evidence.provenance import classify_source_quality
from backend.app.scoring.evidence_verdict import reconcile
from backend.app.analysis import pipeline as analysis
from backend.app.evidence.models import RetrievedEvidence


def factor(score,provenance):
    return {
        "evidence_strength":{"score":score,"justification":"Synthetic direct evidence."},
        "factual_consistency":{"score":100,"justification":"Synthetic"},
        "source_provenance":{"score":provenance,"justification":"Synthetic"},
    }


def test_compound_separate_queries_reserved():
    claim="The Moon rotates on its axis, and global mean sea level has increased over recent decades."
    facets=atomic_search_facets(claim)
    assert len(facets)==2
    queries=plan_verification_queries(claim,4)
    assert any("Moon rotates" in q and "sea level" not in q for q in queries)
    assert any("sea level" in q and "Moon" not in q for q in queries)
    assert claim in queries


def test_original_publication_queries_for_study_or_statement():
    study="A 2023 Cochrane review conclusively proved that face masks never reduce the transmission of respiratory viruses."
    attributed="In March 2023, Cochrane stated that interpreting its mask review as proof that masks do not work was inaccurate."
    for claim in (study,attributed):
        queries=plan_verification_queries(claim,4)
        assert claim in queries
        assert any("original" in q and ("conclusion" in q or "statement" in q) for q in queries)
        assert any("Cochrane" in q for q in queries)


def test_supported_claim_score_cannot_fall_in_low_credibility_band_due_to_weak_sources():
    for value in (58,63):
        factors=factor(value,40)
        score,reason=reconcile("supports", factors)
        assert reason is None
        assert 70<=score<=79
        assert factors["factual_consistency"]["score"]==100
    assert reconcile("supports",factor(100,95))[0] >= 80
    assert reconcile("contradicts",factor(9,95))[0] <= 49


def test_original_publisher_high_authority_and_social_wall():
    assert classify_source_quality("https://www.cochrane.org/news/test") == "high"
    assert is_access_wall("Log in or sign up to view photos")
    assert is_access_wall("Please enable JavaScript to continue")
    assert not is_access_wall("The Moon rotates on its axis in the same time as its orbital period")


def test_ambiguous_bare_comparison_without_metric_is_not_assumed():
    out=analysis._prepare_normal_claim("London is larger than Paris.")
    assert not out["is_clear"]
    assert "underspecified" in out["reason"]
    assert analysis._prepare_normal_claim("London has a larger population than Paris.")["is_clear"]


def test_recovery_queries_target_unresolved_facets_not_certain_verdicts():
    claim="The Moon rotates on its axis, and global mean sea level has increased over recent decades."
    first={"label":"Insufficient Evidence", "evidence_analysis":{
        "subclaim_assessments":[
            {"subclaim":"The Moon rotates on its axis", "status":"insufficient","coverage":"none"},
            {"subclaim":"Global mean sea level has increased", "status":"supports","coverage":"complete"}]}}
    q=recovery_query(claim, first)
    assert "Moon rotates" in q
    assert "sea level" not in q
    assert recovery_query(claim,{"label":"Highly Credible", "evidence_analysis":{}}) is None


def test_normal_retries_after_insufficient_when_new_source_available(monkeypatch):
    claim="The Moon rotates on its axis, and global mean sea level has increased over recent decades."
    sea=RetrievedEvidence(url="https://climate.gov/sea",source="climate.gov",title="Sea level",quality="high",text="Sea level has increased")
    lunar=RetrievedEvidence(url="https://science.nasa.gov/moon",source="science.nasa.gov",title="Moon rotation",quality="high",text="The Moon rotates on its axis")
    search_calls=[]
    def retrieve(**kw):
        search_calls.append(kw)
        return [sea] if len(search_calls)==1 else [lunar]
    analysis_calls=[]
    def evaluate(**kw):
        analysis_calls.append(kw)
        if len(analysis_calls)==1:
            return {"label":"Insufficient Evidence", "budget_exhausted":False, "flags":[],
                    "evidence_analysis":{"subclaim_assessments":[{"subclaim":"The Moon rotates on its axis","status":"insufficient","coverage":"none"}]}}
        return {"label":"Highly Credible", "budget_exhausted":False,"overall_score":90,"flags":[],
                "evidence_analysis":{"subclaim_assessments":[]}}
    monkeypatch.setattr(analysis,"search_and_retrieve",retrieve)
    monkeypatch.setattr(analysis,"analyze_claim_with_llm",evaluate)
    out=analysis.analyze_claim(claim,mode="normal")
    assert len(search_calls)==2
    assert len(analysis_calls)==2
    assert search_calls[1]["query"]==search_calls[1]["alternate_query"]
    assert out["label"]=="Highly Credible"
    assert out["retrieval_recovery"]["new_source_count"]==1
    assert len(out["evidence"])==2

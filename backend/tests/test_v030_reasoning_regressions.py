"""Offline v0.3.0 regressions based on observed structural AVeriTeC failures.

Synthetic passages test control flow, NOT real-world truth or measured accuracy.
"""
from backend.app.evidence.conflict_verdict import grounded_mixed_evidence
from backend.app.evidence.models import RetrievedEvidence
from backend.app.evidence.quality_policy import directional_gate
from backend.app.llm import pipeline
from backend.app.llm.evidence_analysis import apply_v4_subclaim_consistency
from backend.app.models.schemas import AnalysisResponse
from backend.app.retrieval.verification_plan import plan_verification_queries
from backend.app.scoring.uncertainty import evaluate_uncertainty
from backend.app.analysis.pipeline import _prepare_normal_claim
from eval.metrics.label_mapping import trustnet_operational_to_averitec


def doc(domain, quality="medium"):
    return RetrievedEvidence(title="Fact-check", url=f"https://{domain}/claim",
                             source=domain, text="Synthetic fixture only.", quality=quality)


def row(document, stance, qualifier="matches", relevance="high", directness="direct"):
    return dict(title=document.title,url=document.url,stance=stance,quality=document.quality,
                relevance=relevance,directness=directness,qualifier_relation=qualifier,
                reasoning="Synthetic direct material passage.")


def sub(status="mixed", context=False):
    return dict(subclaim="Material claim",status=status,coverage="complete",
                direct_support=status in {"supports","mixed"},
                direct_contradiction=status in {"contradicts","mixed"},
                material_context_conflict=context,reasoning="Fixture")


def test_mixed_direct_conflicting_evidence_is_not_automatically_nei(monkeypatch):
    # Replays the structural failure in AVeriTeC claim #324, not its exact facts.
    repeating=doc("repeated-claim.example.org", "low")
    investigation=doc("factcheck.org", "medium")
    evidence=[repeating,investigation]
    analysis=dict(claim="Synthetic misleading attribution",
                  summary="Repeated allegation and documented incompatible timing.",
                  evidence_assessments=[row(repeating,"supports"),row(investigation,"contradicts",qualifier="conflicts")],
                  subclaim_assessments=[sub("mixed", True)],overall_evidence_direction="mixed")
    assert grounded_mixed_evidence(analysis,evidence)["grounded"]
    assert directional_gate("mixed",analysis["evidence_assessments"],evidence,None,2,analysis["subclaim_assessments"]) is None
    monkeypatch.setattr(pipeline,"timed_evidence_analysis",lambda *args,**kw: (analysis,1))
    monkeypatch.setattr(pipeline,"timed_credibility_scoring",lambda *args,**kw:(dict(claim="Synthetic misleading attribution",flags=[],factors={
        key:dict(score=50,justification="Test only") for key in
        ("factual_consistency","source_provenance","bias_framing","incentive_conflict")}),1))
    result=pipeline.analyze_claim_with_llm("Synthetic misleading attribution",evidence)
    assert result["label"] == "Conflicting Evidence/Cherrypicking"
    assert result["overall_score"] is None
    assert not result["uncertainty"]["gate_triggered"]
    assert trustnet_operational_to_averitec(result)=="Conflicting Evidence/Cherrypicking"


def test_weak_opposing_opinions_do_not_make_valid_conflict():
    a=doc("a.blog.example","low");b=doc("b.blog.example","low")
    assessment=dict(evidence_assessments=[row(a,"supports"),row(b,"contradicts")],
                    subclaim_assessments=[sub("mixed")])
    assert not grounded_mixed_evidence(assessment,[a,b])["grounded"]
    gate,_,_=evaluate_uncertainty(2,0,True,True,"mixed",0,False)
    assert gate


def test_direct_qualifier_conflict_requires_reliable_passage():
    a=doc("factcheck.org")
    info=dict(evidence_assessments=[row(a,"supports",qualifier="matches")],
              subclaim_assessments=[sub("mixed",True)])
    assert not grounded_mixed_evidence(info,[a])["grounded"]
    assert not grounded_mixed_evidence(info,[a])["material_context"]


def test_mixed_source_with_direct_authoritative_context_is_valid():
    a=doc("factcheck.org")
    info=dict(evidence_assessments=[row(a,"mixed",qualifier="conflicts")],
              subclaim_assessments=[sub("mixed",True)])
    assert grounded_mixed_evidence(info,[a])["grounded"]


def test_unfounded_context_flag_is_dropped_before_direction_selection():
    a=doc("factcheck.org")
    analysis=dict(evidence_assessments=[row(a,"supports")],
                  subclaim_assessments=[sub("supports",True)])
    apply_v4_subclaim_consistency(analysis)
    assert analysis["subclaim_assessments"][0]["status"]=="supports"
    assert analysis["subclaim_assessments"][0]["material_context_conflict"] is False


def test_quote_context_and_historical_year_in_queries():
    claim="Public official said he wants to ban fracking."
    q=plan_verification_queries(claim,4)
    assert any("transcript" in x for x in q)
    assert claim in q
    context="Speaker: official\nClaim date: 02-10-2020\nLocation: US"
    p=_prepare_normal_claim(claim,context,mode="normal")
    allq=[p["search_query"]]+p["verification_queries"]
    assert any("2020" in x for x in allq)
    assert any(claim in x for x in allq)


def test_comparison_queries_still_cover_both_facets():
    q=plan_verification_queries("Sweden has lower population than India but a higher happiness index.",4)
    assert any("population statistics" in x for x in q)
    assert any("happiness report" in x for x in q)


def test_public_schema_accepts_distinct_conflicting_label():
    kwargs=dict(claim="Claim",overall_score=None,label="Conflicting Evidence/Cherrypicking",
                confidence="medium",factors=None,
                uncertainty=dict(gate_triggered=False,level="medium",reason="Context documented"),
                evidence=[],flags=[],latency_ms=100)
    assert AnalysisResponse(**kwargs).label=="Conflicting Evidence/Cherrypicking"

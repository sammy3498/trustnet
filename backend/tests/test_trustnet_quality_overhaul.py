"""Offline regression tests: no credentials or live searches required."""
from copy import deepcopy
from types import SimpleNamespace as NS

from backend.app.evidence.models import RetrievedEvidence
from backend.app.evidence.direction import resolve_overall_evidence_direction
from backend.app.evidence.provenance import classify_source_quality
from backend.app.llm.evidence_analysis import apply_retrieved_source_quality, enforce_subclaim_grounding
from backend.app.llm import pipeline as pipeline
from backend.app.retrieval.source_priority import prioritize_for_fetch, select_facet_evidence
from backend.app.retrieval.verification_plan import plan_verification_queries
from backend.app.api.routes import build_public_evidence


def doc(url, title, text, quality='high', origin='webpage'):
    return RetrievedEvidence(
        title=title, url=url, source=url.split('/')[2], text=text,
        quality=quality, evidence_origin=origin,
    )


def row(d, stance, direct='direct', relevance='high'):
    return dict(title=d.title, url=d.url, quality=d.quality, stance=stance,
                directness=direct, relevance=relevance, evidence_origin=d.evidence_origin,
                reasoning='Direct textual comparison of the material fact.')


def subclaim(name, status, coverage='complete'):
    return dict(subclaim=name, status=status, coverage=coverage, direct_support=status=='supports',
                direct_contradiction=status=='contradicts', material_context_conflict=False, reasoning='fixture')


def llm_factors(factual=50):
    return dict(claim='fixture', flags=[], factors={
        k: dict(score=s, justification='Synthetic secondary factor value')
        for k,s in {'factual_consistency': factual, 'source_provenance': 50,
                     'bias_framing': 95, 'incentive_conflict': 95}.items()})


def replay(monkeypatch, claim, evidence, rows, subs, llm_fact=50):
    direction=resolve_overall_evidence_direction(subs, 'insufficient', claim_text=claim)
    analysis = dict(claim=claim, evidence_assessments=rows, subclaim_assessments=subs,
                    overall_evidence_direction=direction, summary='Evidence-verified fixture.')
    monkeypatch.setattr(pipeline, 'timed_evidence_analysis', lambda *args,**kw: (analysis, 1))
    monkeypatch.setattr(pipeline, 'timed_credibility_scoring', lambda *args,**kw: (llm_factors(llm_fact), 1))
    return pipeline.analyze_claim_with_llm(claim=claim, evidence=evidence, mode='normal')


def test_population_and_happiness_have_separate_retrieval_facets():
    queries = plan_verification_queries('Sweden has a smaller population than India but has a higher happiness index.', 4)
    assert any('population statistics' in q for q in queries)
    assert any('happiness report' in q for q in queries)
    assert any('Sweden has a smaller population' in q for q in queries)


def test_voting_abstract_universal_query_looks_for_exceptions():
    queries = plan_verification_queries('No one under 18 can vote in any U.S. election.', 4)
    assert any('exceptions' in q for q in queries)
    assert any('voter eligibility' in q for q in queries)


def test_subject_domains_are_not_spoofable():
    assert classify_source_quality('https://worldhappiness.report/ed/2026/') == 'high'
    assert classify_source_quality('https://ncsl.org/elections-and-campaigns/voting-age') == 'high'
    assert classify_source_quality('https://usvotefoundation.org/voter-help-desk') == 'medium'
    assert classify_source_quality('https://ncsl.org.attacker.example/fake') == 'low'


def test_voting_single_authoritative_counterexample_overrides_bad_secondary_score(monkeypatch):
    high = doc('https://www.ncsl.org/elections-and-campaigns/voting-age-for-primary-elections',
               'Voting Age for Primary Elections', 'Some states allow eligible 17-year-olds to vote in primaries.')
    irrelevant = doc('https://www.whitehouse.gov/example','Federal elections','Voters register to vote.')
    result = replay(monkeypatch, 'No one under 18 can vote in any U.S. election.',
                    [irrelevant, high], [row(irrelevant,'neutral'),row(high,'contradicts')],
                    [subclaim('No person under 18 can vote in any U.S. election','contradicts')],llm_fact=100)
    assert result['label'] == 'Not Credible'
    assert result['overall_score'] <= 20
    assert result['uncertainty']['gate_triggered'] is False
    assert result['factors']['factual_consistency']['score']==0


def test_sweden_compound_supported_by_independent_original_sources(monkeypatch):
    pop = doc('https://data.worldbank.org/indicator/SP.POP.TOTL','World Bank population',
              'India population 1,400 million. Sweden population 11 million.')
    happiness = doc('https://www.worldhappiness.report/ed/2026/','World Happiness Report country rankings',
                    'Sweden ranked fifth; India ranked 116th in 2026.')
    result=replay(monkeypatch, 'Sweden has a smaller population than India but a higher happiness index.',
                  [pop,happiness], [row(pop,'supports','partial'),row(happiness,'supports','partial')],
                  [subclaim('Sweden has a smaller population than India','supports'),
                   subclaim('Sweden ranks higher than India in happiness','supports')],llm_fact=15)
    assert result['label'] == 'Highly Credible'
    assert result['overall_score'] >= 80
    assert not result['uncertainty']['gate_triggered']
    assert result['factors']['factual_consistency']['score']==100


def test_both_true_and_false_conjunct_results_in_false_full_claim(monkeypatch):
    a=doc('https://example.org/a','first source','x is established',quality='medium')
    b=doc('https://www.ncsl.org/elections','second source','y is refuted')
    result=replay(monkeypatch,'Claim X is true but claim Y is true.',[a,b],
                  [row(a,'supports'),row(b,'contradicts')],
                  [subclaim('Claim X','supports'),subclaim('Claim Y','contradicts')])
    assert result['label']=='Not Credible'
    assert result['uncertainty']['gate_triggered'] is False
    assert 'Compound claim' in ' '.join(result['flags'])


def test_one_low_quality_direct_assertion_does_not_get_certified(monkeypatch):
    weak=doc('https://blog.example.org/post','unverified post','a claim',quality='low')
    result=replay(monkeypatch,'Earth orbits the Sun.',[weak],[row(weak,'supports')],
                  [subclaim('Earth orbits the Sun','supports')])
    assert result['label']=='Insufficient Evidence'
    assert result['uncertainty']['gate_triggered']


def test_unrecognized_hallucinated_citation_is_discarded():
    d=doc('https://www.ncsl.org/actual','Actual','An actual source')
    analysis=dict(evidence_assessments=[row(d,'supports'),
             dict(title='Fabricated source',url='https://example.org/made-up',quality='high',
                  stance='supports',directness='direct',relevance='high')],
                  subclaim_assessments=[subclaim('x','supports')])
    out=apply_retrieved_source_quality(analysis,[d])
    assert len(out['evidence_assessments'])==1
    assert out['unmatched_citation_count']==1
    missing=dict(evidence_assessments=[dict(title='invented', url='https://bad.test',stance='supports',
               directness='direct', relevance='high')],subclaim_assessments=[subclaim('x','supports')])
    enforce_subclaim_grounding(missing)
    assert not missing['subclaim_assessments'][0]['direct_support'] or missing['evidence_assessments']


def test_fetch_budget_protects_claim_facets_and_prefers_original_sources():
    # First facet returns two blogs and a source; second returns a primary report.
    docs=[NS(url='https://a.test/1',title='Blog about population',snippet='Sweden India population',quality='low'),
          NS(url='https://data.worldbank.org/pop',title='Country population values',snippet='Sweden India populations',quality='high'),
          NS(url='https://www.worldhappiness.report/2026',title='World happiness rankings',snippet='Sweden India rank',quality='high')]
    q1='Sweden India population statistics'
    q2='Sweden India world happiness rankings'
    ranks={docs[0].url:{q1:1},docs[1].url:{q1:2},docs[2].url:{q2:1}}
    chosen=prioritize_for_fetch(docs,ranks,[q1,q2],2)
    assert {d.url for d in chosen}=={docs[1].url,docs[2].url}


def test_source_list_presents_direct_citation_before_offtopic_government_page():
    relevant=doc('https://www.ncsl.org/voting','Voting Age in Primaries','17-year-olds may vote in some elections')
    background=doc('https://www.whitehouse.gov/citizenship','Citizenship','General background')
    result=dict(evidence=[background.model_dump(),relevant.model_dump()],
                evidence_analysis=dict(evidence_assessments=[row(background,'neutral'),row(relevant,'contradicts')]))
    cited=build_public_evidence(result)
    assert cited[0].url==relevant.url

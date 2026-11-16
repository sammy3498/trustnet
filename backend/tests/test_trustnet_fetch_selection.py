"""Simulate Tavily/fetch/rank without accessing the network or an LLM."""
from backend.app.evidence.models import RetrievedEvidence
from backend.app.retrieval.search_provider import SearchResult
from backend.app.retrieval import search_pipeline as search
from backend.app.retrieval.verification_plan import plan_verification_queries


def test_full_search_retains_one_authoritative_page_per_metric(monkeypatch):
    claim='Sweden has a smaller population than India but has a higher happiness index.'
    queries=plan_verification_queries(claim,4)
    population,happiness=queries[:2]
    original=SearchResult(title='World population numbers', url='https://data.worldbank.org/indicator/SP.POP.TOTL',
                          snippet='Sweden India population statistics')
    report=SearchResult(title='World Happiness Report rankings', url='https://www.worldhappiness.report/ed/2026/',
                        snippet='Sweden India country rankings')
    blog=SearchResult(title='Sweden versus India',url='https://randomblog.example/happiness',
                      snippet='Sweden India population and happiness general discussion')
    def fake_search(query,max_results,end_date=None):
        if query == population:
            return [blog, original]
        if query == happiness:
            return [blog,report]
        return [blog]
    monkeypatch.setattr(search,'run_search',fake_search)

    def make_evidence(url,title=''):
        text=({'https://data.worldbank.org/indicator/SP.POP.TOTL':'Sweden population is 11 million, India 1400 million',
               'https://www.worldhappiness.report/ed/2026/':'World Happiness Report Sweden fifth and India 116th'}
              .get(url,'Blog generally discusses Sweden and India.'))
        quality={'https://data.worldbank.org/indicator/SP.POP.TOTL':'high',
                 'https://www.worldhappiness.report/ed/2026/':'high'}.get(url,'low')
        return RetrievedEvidence(url=url,title=title,source=url.split('/')[2],text=text,quality=quality)
    monkeypatch.setattr(search,'build_evidence_from_url', make_evidence)
    monkeypatch.setattr(search,'build_evidence_from_snippet',lambda url,title,snippet: make_evidence(url,title))
    # Deliberately return a poor ordering from the hybrid ranker.
    monkeypatch.setattr(search,'retrieve_hybrid',lambda **kw: kw['documents'])
    found=search.search_and_retrieve(
        query=claim, alternate_query=population,
        verification_queries=queries[1:], top_k=3, search_results=3,
        budget=None,mode='normal')
    selected={e.url for e in found}
    assert original.url in selected
    assert report.url in selected
    assert all(item.matched_query for item in found)

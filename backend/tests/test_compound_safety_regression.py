from backend.app.analysis import pipeline
from backend.app.evidence import (
    compound_fast_path,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)


def evidence(
    url: str,
    source: str,
    quality: str = "medium",
    text: str = "Relevant evidence.",
    evidence_origin: str = "webpage",
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=source,
        url=url,
        source=source,
        text=text,
        quality=quality,
        evidence_origin=evidence_origin,
    )


def fast_analysis(
    claim: str,
    direction: str,
    url: str,
) -> dict:
    supports = (
        direction == "supports"
    )

    return {
        "claim": claim,
        "evidence_assessments": [
            {
                "title": "Evidence",
                "url": url,
                "stance": direction,
                "quality": "high",
                "relevance": "high",
                "directness": "direct",
                "qualifier_relation": (
                    "not_applicable"
                ),
                "reasoning": "Resolved.",
            }
        ],
        "subclaim_assessments": [
            {
                "subclaim": claim,
                "coverage": "complete",
                "direct_support": supports,
                "direct_contradiction": (
                    not supports
                ),
                "material_context_conflict": False,
                "status": direction,
                "reasoning": "Resolved.",
            }
        ],
        "overall_evidence_direction": (
            direction
        ),
        "summary": "Resolved.",
        "bias_framing": "neutral",
        "bias_framing_justification": (
            "No material framing concern."
        ),
        "incentive_conflict": (
            "none_detected"
        ),
        "incentive_conflict_justification": (
            "No incentive conflict."
        ),
        "material_caveats": [],
        "_fast_path": {
            "effective_evidence_count": 2,
            "effective_high_quality_count": 1,
            "effective_medium_quality_count": 1,
        },
    }


def test_recovered_only_probe_beats_initial_contamination(
    monkeypatch,
):
    contaminant_url = (
        "https://en.wikipedia.org/wiki/Moon"
    )

    recovered_urls = {
        (
            "https://science.nasa.gov/"
            "earth/orbit/"
        ),
        (
            "https://example.edu/"
            "earth-sun/"
        ),
    }

    initial = [
        evidence(
            url=contaminant_url,
            source="en.wikipedia.org",
            quality="low",
            text=(
                "The Moon orbits the Sun "
                "in a heliocentric frame."
            ),
            evidence_origin="snippet",
        )
    ]

    recovered = [
        evidence(
            url=(
                "https://science.nasa.gov/"
                "earth/orbit/"
            ),
            source="science.nasa.gov",
            quality="high",
            text=(
                "Earth orbits the Sun."
            ),
        ),
        evidence(
            url=(
                "https://example.edu/"
                "earth-sun/"
            ),
            source="example.edu",
            quality="medium",
            text=(
                "Earth travels around "
                "the Sun."
            ),
        ),
    ]

    probe_url_sets = []

    def fake_fast(
        claim,
        evidence,
    ):
        urls = {
            item.url
            for item in evidence
        }

        probe_url_sets.append(
            urls
        )

        if urls == recovered_urls:
            return fast_analysis(
                claim=claim,
                direction="supports",
                url=(
                    "https://science.nasa.gov/"
                    "earth/orbit/"
                ),
            )

        return None

    def fake_recovery(
        claim,
        existing_evidence,
        end_date=None,
        deadline=None,
    ):
        return list(
            recovered
        )

    monkeypatch.setattr(
        compound_fast_path,
        "try_fast_evidence_analysis",
        fake_fast,
    )

    monkeypatch.setattr(
        compound_fast_path,
        "recover_authority_evidence",
        fake_recovery,
    )

    result = (
        compound_fast_path
        .resolve_atomic_subclaim(
            subclaim=(
                "The Earth orbits the Sun."
            ),
            evidence=list(
                initial
            ),
        )
    )

    assert (
        result[
            "analysis"
        ][
            "overall_evidence_direction"
        ]
        == "supports"
    )

    assert len(
        probe_url_sets
    ) == 2

    assert (
        contaminant_url
        in probe_url_sets[
            0
        ]
    )

    assert (
        probe_url_sets[
            1
        ]
        == recovered_urls
    )


def test_resolved_compound_never_calls_terra(
    monkeypatch,
):
    claim = (
        "Earth is round and "
        "Mars is red."
    )

    subclaims = [
        "Earth is round.",
        "Mars is red.",
    ]

    initial_evidence = [
        evidence(
            url="https://a.edu/earth",
            source="a.edu",
            quality="medium",
        ),
        evidence(
            url="https://b.edu/mars",
            source="b.edu",
            quality="medium",
        ),
    ]

    earth = fast_analysis(
        claim=subclaims[
            0
        ],
        direction="supports",
        url="https://a.edu/earth",
    )

    mars = fast_analysis(
        claim=subclaims[
            1
        ],
        direction="supports",
        url="https://b.edu/mars",
    )

    compound_analysis = (
        compound_fast_path
        .combine_subclaim_analyses(
            claim=claim,
            subclaims=subclaims,
            analyses=[
                earth,
                mars,
            ],
        )
    )

    assert compound_analysis is not None

    def fake_prepare_claim(
        input_claim,
        context=None,
    ):
        return {
            "normalized_claim": claim,
            "is_clear": True,
            "reason": (
                "The claim is objectively "
                "verifiable."
            ),
            "search_query": claim,
            "verification_queries": (
                subclaims
            ),
            "timed_out": False,
        }

    def fake_search(
        *args,
        **kwargs,
    ):
        return list(
            initial_evidence
        )

    def fake_compound(
        *args,
        **kwargs,
    ):
        return {
            "analysis": (
                compound_analysis
            ),
            "recovered_evidence": [],
            "fast_path_ms": 3,
            "authority_recovery_ms": 0,
        }

    def fake_score(
        **kwargs,
    ):
        evidence_analysis = (
            kwargs[
                "evidence_analysis"
            ]
        )

        return {
            "claim": claim,
            "overall_score": 99,
            "label": "Highly Credible",
            "factors": {
                "evidence_strength": {
                    "score": 99,
                    "justification": "Test.",
                },
                "factual_consistency": {
                    "score": 99,
                    "justification": "Test.",
                },
                "source_provenance": {
                    "score": 100,
                    "justification": "Test.",
                },
                "bias_framing": {
                    "score": 100,
                    "justification": "Test.",
                },
                "incentive_conflict": {
                    "score": 100,
                    "justification": "Test.",
                },
            },
            "flags": [],
            "uncertainty": {
                "gate_triggered": False,
                "level": "low",
                "reason": (
                    "Sufficient consistent "
                    "evidence is available."
                ),
            },
            "evidence_analysis": (
                evidence_analysis
            ),
            "analysis_degraded": False,
            "analysis_path": (
                "local_nli_compound"
            ),
        }

    def terra_must_not_run(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            "Terra/LLM fallback was called "
            "after successful compound "
            "resolution."
        )

    monkeypatch.setattr(
        pipeline,
        "prepare_claim",
        fake_prepare_claim,
    )

    monkeypatch.setattr(
        pipeline,
        "search_and_retrieve",
        fake_search,
    )

    monkeypatch.setattr(
        pipeline,
        "is_simple_atomic_claim",
        lambda value: (
            " and "
            not in value.lower()
        ),
        raising=False,
    )

    monkeypatch.setattr(
        pipeline,
        "try_compound_fast_evidence_analysis",
        fake_compound,
        raising=False,
    )

    monkeypatch.setattr(
        pipeline,
        "score_evidence_analysis",
        fake_score,
        raising=False,
    )

    monkeypatch.setattr(
        pipeline,
        "analyze_claim_with_llm",
        terra_must_not_run,
    )

    if hasattr(
        pipeline,
        "get_cached_analysis",
    ):
        monkeypatch.setattr(
            pipeline,
            "get_cached_analysis",
            lambda *args, **kwargs: None,
        )

    if hasattr(
        pipeline,
        "set_cached_analysis",
    ):
        monkeypatch.setattr(
            pipeline,
            "set_cached_analysis",
            lambda *args, **kwargs: None,
        )

    result = (
        pipeline.analyze_claim(
            claim,
            search_results=5,
            top_k=3,
        )
    )

    assert (
        result[
            "analysis_path"
        ]
        == "local_nli_compound"
    )

    assert (
        result[
            "evidence_analysis"
        ][
            "overall_evidence_direction"
        ]
        == "supports"
    )

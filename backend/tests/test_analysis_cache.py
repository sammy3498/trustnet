from backend.app.analysis.cache import (
    build_analysis_cache_key,
    clear_analysis_cache,
    get_cached_analysis,
    set_cached_analysis,
)
from backend.app.analysis.pipeline import (
    analyze_claim,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)


def setup_function():
    clear_analysis_cache()


def teardown_function():
    clear_analysis_cache()


def test_cache_key_normalizes_claim_text():
    first = (
        build_analysis_cache_key(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            search_results=5,
            top_k=3,
        )
    )

    second = (
        build_analysis_cache_key(
            claim=(
                "  the   earth "
                "ORBITS the sun.  "
            ),
            search_results=5,
            top_k=3,
        )
    )

    assert first == second


def test_cache_returns_deep_copy():
    key = (
        build_analysis_cache_key(
            claim="Test claim",
            search_results=5,
            top_k=3,
        )
    )

    original = {
        "overall_score": 99,
        "flags": [],
    }

    set_cached_analysis(
        key,
        original,
    )

    cached = (
        get_cached_analysis(
            key
        )
    )

    cached[
        "flags"
    ].append(
        "Changed"
    )

    second = (
        get_cached_analysis(
            key
        )
    )

    assert second[
        "flags"
    ] == []


def test_repeated_analysis_uses_cache(
    monkeypatch,
):
    calls = {
        "prepare": 0,
        "retrieve": 0,
        "analyze": 0,
    }

    def fake_prepare(
        claim,
        context=None,
    ):
        calls[
            "prepare"
        ] += 1

        return {
            "normalized_claim": (
                claim
            ),
            "is_clear": True,
            "reason": (
                "Clear claim."
            ),
            "search_query": (
                claim
            ),
            "verification_queries": [],
            "timed_out": False,
        }

    evidence = (
        RetrievedEvidence(
            title="NASA",
            url=(
                "https://nasa.gov/test"
            ),
            source="nasa.gov",
            text="Evidence.",
            quality="high",
        )
    )

    def fake_retrieve(
        **kwargs,
    ):
        calls[
            "retrieve"
        ] += 1

        return [
            evidence
        ]

    def fake_analysis(
        claim,
        evidence,
    ):
        calls[
            "analyze"
        ] += 1

        return {
            "claim": claim,
            "overall_score": 99,
            "label": (
                "Highly Credible"
            ),
            "factors": {
                "evidence_strength": {
                    "score": 99,
                    "justification": (
                        "Strong evidence."
                    ),
                },
                "factual_consistency": {
                    "score": 99,
                    "justification": (
                        "Consistent."
                    ),
                },
                "source_provenance": {
                    "score": 100,
                    "justification": (
                        "Strong provenance."
                    ),
                },
                "bias_framing": {
                    "score": 100,
                    "justification": (
                        "Neutral."
                    ),
                },
                "incentive_conflict": {
                    "score": 100,
                    "justification": (
                        "None detected."
                    ),
                },
            },
            "flags": [],
            "uncertainty": {
                "gate_triggered": False,
                "level": "low",
                "reason": (
                    "Sufficient evidence."
                ),
            },
            "evidence_analysis": {
                "evidence_assessments": [],
            },
            "analysis_degraded": False,
        }

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.prepare_claim"
        ),
        fake_prepare,
    )

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.search_and_retrieve"
        ),
        fake_retrieve,
    )

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.analyze_claim_with_llm"
        ),
        fake_analysis,
    )

    first = analyze_claim(
        claim=(
            "The Earth orbits "
            "the Sun."
        ),
        search_results=5,
        top_k=3,
    )

    second = analyze_claim(
        claim=(
            "  the earth   orbits "
            "the SUN. "
        ),
        search_results=5,
        top_k=3,
    )

    assert first[
        "overall_score"
    ] == 99

    assert second[
        "overall_score"
    ] == 99

    assert first[
        "cache"
    ][
        "hit"
    ] is False

    assert second[
        "cache"
    ][
        "hit"
    ] is True

    assert calls == {
        "prepare": 1,
        "retrieve": 1,
        "analyze": 1,
    }


def test_degraded_analysis_is_not_cached(
    monkeypatch,
):
    calls = {
        "analyze": 0,
    }

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.prepare_claim"
        ),
        lambda claim, context=None: {
            "normalized_claim": (
                claim
            ),
            "is_clear": True,
            "reason": "Clear.",
            "search_query": claim,
            "verification_queries": [],
            "timed_out": False,
        },
    )

    evidence = (
        RetrievedEvidence(
            title="Evidence",
            url=(
                "https://nasa.gov/test"
            ),
            source="nasa.gov",
            text="Evidence.",
            quality="high",
        )
    )

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.search_and_retrieve"
        ),
        lambda **kwargs: [
            evidence
        ],
    )

    def fake_analysis(
        claim,
        evidence,
    ):
        calls[
            "analyze"
        ] += 1

        return {
            "claim": claim,
            "overall_score": 90,
            "label": (
                "Highly Credible"
            ),
            "factors": {
                "evidence_strength": {
                    "score": 95,
                    "justification": "Test.",
                },
                "factual_consistency": {
                    "score": 99,
                    "justification": "Test.",
                },
                "source_provenance": {
                    "score": 90,
                    "justification": "Test.",
                },
                "bias_framing": {
                    "score": 85,
                    "justification": "Test.",
                },
                "incentive_conflict": {
                    "score": 85,
                    "justification": "Test.",
                },
            },
            "flags": [],
            "uncertainty": {
                "gate_triggered": False,
                "level": "low",
                "reason": "Test.",
            },
            "evidence_analysis": {
                "evidence_assessments": [],
            },
            "analysis_degraded": True,
        }

    monkeypatch.setattr(
        (
            "backend.app.analysis."
            "pipeline.analyze_claim_with_llm"
        ),
        fake_analysis,
    )

    analyze_claim(
        "Test claim",
        search_results=5,
        top_k=3,
    )

    analyze_claim(
        "Test claim",
        search_results=5,
        top_k=3,
    )

    assert calls[
        "analyze"
    ] == 2
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.pipeline import (
    analyze_claim_with_llm,
)


def make_evidence(
    url: str,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title="Evidence",
        url=url,
        source=url,
        text="The Earth orbits the Sun.",
        quality="high",
    )


def test_single_analysis_produces_deterministic_99(
    monkeypatch,
):
    evidence = [
        make_evidence(
            "https://nasa.gov/a"
        ),
        make_evidence(
            "https://who.int/b"
        ),
    ]

    fake_analysis = {
        "claim": (
            "The Earth orbits the Sun."
        ),
        "evidence_assessments": [
            {
                "title": "NASA",
                "url": (
                    "https://nasa.gov/a"
                ),
                "stance": "supports",
                "quality": "high",
                "relevance": "high",
                "directness": "direct",
                "qualifier_relation": (
                    "matches"
                ),
                "reasoning": (
                    "Direct support."
                ),
            },
            {
                "title": "WHO",
                "url": (
                    "https://who.int/b"
                ),
                "stance": "supports",
                "quality": "high",
                "relevance": "high",
                "directness": "direct",
                "qualifier_relation": (
                    "matches"
                ),
                "reasoning": (
                    "Independent support."
                ),
            },
        ],
        "subclaim_assessments": [
            {
                "subclaim": (
                    "The Earth orbits "
                    "the Sun."
                ),
                "coverage": "complete",
                "direct_support": True,
                "direct_contradiction": (
                    False
                ),
                "material_context_conflict": (
                    False
                ),
                "status": "supports",
                "reasoning": (
                    "Fully supported."
                ),
            }
        ],
        "overall_evidence_direction": (
            "supports"
        ),
        "summary": (
            "The evidence supports "
            "the claim."
        ),
        "bias_framing": "neutral",
        "bias_framing_justification": (
            "No material framing issue."
        ),
        "incentive_conflict": (
            "none_detected"
        ),
        "incentive_conflict_justification": (
            "No material conflict."
        ),
        "material_caveats": [],
    }

    monkeypatch.setattr(
        (
            "backend.app.llm.pipeline."
            "analyze_evidence_with_llm"
        ),
        lambda claim, evidence: (
            fake_analysis
        ),
    )

    result = (
        analyze_claim_with_llm(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            evidence=evidence,
        )
    )

    assert (
        result[
            "overall_score"
        ]
        == 99
    )

    assert (
        result[
            "label"
        ]
        == "Highly Credible"
    )

    assert (
        result[
            "factors"
        ][
            "evidence_strength"
        ][
            "score"
        ]
        == 99
    )

    assert (
        result[
            "factors"
        ][
            "factual_consistency"
        ][
            "score"
        ]
        == 99
    )

    assert (
        result[
            "factors"
        ][
            "source_provenance"
        ][
            "score"
        ]
        == 100
    )

    assert (
        result[
            "flags"
        ]
        == []
    )

    assert (
        result[
            "analysis_degraded"
        ]
        is False
    )
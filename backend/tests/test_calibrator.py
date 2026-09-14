from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.scoring.calibrator import (
    build_calibrated_factors,
    calibrate_evidence_strength,
    calibrate_factual_consistency,
    calibrate_source_provenance,
)


def evidence_item(
    url: str,
    quality: str,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title="Test",
        url=url,
        source=url,
        text="Test evidence.",
        quality=quality,
    )


def supporting_analysis(
    assessments,
):
    return {
        "overall_evidence_direction": (
            "supports"
        ),
        "evidence_assessments": (
            assessments
        ),
        "subclaim_assessments": [
            {
                "subclaim": (
                    "Test claim"
                ),
                "coverage": (
                    "complete"
                ),
                "direct_support": True,
                "direct_contradiction": (
                    False
                ),
                "material_context_conflict": (
                    False
                ),
                "status": "supports",
                "reasoning": (
                    "Direct support."
                ),
            }
        ],
    }


def test_two_independent_high_quality_sources_get_99():
    evidence = [
        evidence_item(
            "https://nasa.gov/a",
            "high",
        ),
        evidence_item(
            "https://who.int/b",
            "high",
        ),
    ]

    analysis = (
        supporting_analysis(
            [
                {
                    "url": (
                        "https://nasa.gov/a"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
                {
                    "url": (
                        "https://who.int/b"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
            ]
        )
    )

    assert (
        calibrate_evidence_strength(
            analysis,
            evidence,
        )
        == 99
    )


def test_single_high_quality_direct_support_gets_95():
    evidence = [
        evidence_item(
            "https://nasa.gov/a",
            "high",
        ),
    ]

    analysis = (
        supporting_analysis(
            [
                {
                    "url": (
                        "https://nasa.gov/a"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                }
            ]
        )
    )

    assert (
        calibrate_evidence_strength(
            analysis,
            evidence,
        )
        == 95
    )


def test_two_low_quality_supports_do_not_get_extreme_score():
    evidence = [
        evidence_item(
            "https://example-one.com/a",
            "low",
        ),
        evidence_item(
            "https://example-two.com/b",
            "low",
        ),
    ]

    analysis = (
        supporting_analysis(
            [
                {
                    "url": (
                        "https://example-one.com/a"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
                {
                    "url": (
                        "https://example-two.com/b"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
            ]
        )
    )

    assert (
        calibrate_evidence_strength(
            analysis,
            evidence,
        )
        == 72
    )


def test_supported_subclaim_has_fixed_factual_consistency():
    analysis = (
        supporting_analysis(
            []
        )
    )

    assert (
        calibrate_factual_consistency(
            analysis
        )
        == 99
    )


def test_two_high_quality_sources_have_max_provenance():
    evidence = [
        evidence_item(
            "https://nasa.gov/a",
            "high",
        ),
        evidence_item(
            "https://who.int/b",
            "high",
        ),
    ]

    analysis = (
        supporting_analysis(
            [
                {
                    "url": (
                        "https://nasa.gov/a"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
                {
                    "url": (
                        "https://who.int/b"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
            ]
        )
    )

    assert (
        calibrate_source_provenance(
            analysis,
            evidence,
        )
        == 100
    )


def test_extreme_support_state_produces_99_overall_components():
    evidence = [
        evidence_item(
            "https://nasa.gov/a",
            "high",
        ),
        evidence_item(
            "https://who.int/b",
            "high",
        ),
    ]

    analysis = (
        supporting_analysis(
            [
                {
                    "url": (
                        "https://nasa.gov/a"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
                {
                    "url": (
                        "https://who.int/b"
                    ),
                    "stance": "supports",
                    "relevance": "high",
                    "directness": "direct",
                },
            ]
        )
    )

    assessment = {
        "bias_framing": (
            "neutral"
        ),
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

    factors = (
        build_calibrated_factors(
            evidence_analysis=(
                analysis
            ),
            credibility_assessment=(
                assessment
            ),
            evidence=evidence,
        )
    )

    assert (
        factors[
            "evidence_strength"
        ]["score"]
        == 99
    )

    assert (
        factors[
            "factual_consistency"
        ]["score"]
        == 99
    )

    assert (
        factors[
            "source_provenance"
        ]["score"]
        == 100
    )

    assert (
        factors[
            "bias_framing"
        ]["score"]
        == 100
    )

    assert (
        factors[
            "incentive_conflict"
        ]["score"]
        == 100
    )
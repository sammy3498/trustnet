from backend.app.evidence import (
    compound_fast_path,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)


def evidence(
    url: str,
    quality: str = "medium",
):
    source = (
        url
        .split(
            "//",
            1,
        )[1]
        .split(
            "/",
            1,
        )[0]
    )

    return RetrievedEvidence(
        title="Evidence",
        url=url,
        source=source,
        text="Relevant evidence.",
        quality=quality,
        evidence_origin="webpage",
    )


def fast_analysis(
    claim: str,
    direction: str,
    url: str,
    quality: str = "medium",
):
    return {
        "claim": claim,
        "evidence_assessments": [
            {
                "title": "Evidence",
                "url": url,
                "stance": direction,
                "quality": quality,
                "relevance": "high",
                "directness": "direct",
                "qualifier_relation": (
                    "not_applicable"
                ),
                "reasoning": (
                    "Direct local evidence."
                ),
            }
        ],
        "subclaim_assessments": [
            {
                "subclaim": claim,
                "coverage": "complete",
                "direct_support": (
                    direction
                    == "supports"
                ),
                "direct_contradiction": (
                    direction
                    == "contradicts"
                ),
                "material_context_conflict": (
                    False
                ),
                "status": direction,
                "reasoning": (
                    "Resolved locally."
                ),
            }
        ],
        "overall_evidence_direction": (
            direction
        ),
        "summary": "Resolved locally.",
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
            "route": "local_nli",
            "effective_evidence_count": 2,
            "effective_high_quality_count": 0,
            "effective_medium_quality_count": 2,
        },
    }


def test_all_supported_compound_is_supported(
    monkeypatch,
):
    analyses = {
        "Earth is round.": (
            fast_analysis(
                "Earth is round.",
                "supports",
                "https://a.edu/earth",
            )
        ),
        "Mars is red.": (
            fast_analysis(
                "Mars is red.",
                "supports",
                "https://b.edu/mars",
            )
        ),
    }

    monkeypatch.setattr(
        compound_fast_path,
        "try_fast_evidence_analysis",
        lambda claim, evidence: (
            analyses[
                claim
            ]
        ),
    )

    result = (
        compound_fast_path
        .try_compound_fast_evidence_analysis(
            claim=(
                "Earth is round and "
                "Mars is red."
            ),
            subclaims=[
                "Earth is round.",
                "Mars is red.",
            ],
            evidence=[
                evidence(
                    "https://a.edu/earth"
                ),
                evidence(
                    "https://b.edu/mars"
                ),
            ],
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
        result[
            "analysis"
        ][
            "subclaim_assessments"
        ]
    ) == 2


def test_one_false_component_contradicts_compound(
    monkeypatch,
):
    analyses = {
        "Earth is round.": (
            fast_analysis(
                "Earth is round.",
                "supports",
                "https://a.edu/earth",
            )
        ),
        "Mars is green.": (
            fast_analysis(
                "Mars is green.",
                "contradicts",
                "https://b.edu/mars",
            )
        ),
    }

    monkeypatch.setattr(
        compound_fast_path,
        "try_fast_evidence_analysis",
        lambda claim, evidence: (
            analyses[
                claim
            ]
        ),
    )

    result = (
        compound_fast_path
        .try_compound_fast_evidence_analysis(
            claim=(
                "Earth is round and "
                "Mars is green."
            ),
            subclaims=[
                "Earth is round.",
                "Mars is green.",
            ],
            evidence=[
                evidence(
                    "https://a.edu/earth"
                ),
                evidence(
                    "https://b.edu/mars"
                ),
            ],
        )
    )

    assert (
        result[
            "analysis"
        ][
            "overall_evidence_direction"
        ]
        == "contradicts"
    )


def test_unresolved_subclaim_causes_fallback(
    monkeypatch,
):
    calls = {
        "fast": 0,
        "recovery": 0,
    }

    def fake_fast(
        claim,
        evidence,
    ):
        calls[
            "fast"
        ] += 1

        if (
            claim
            == "Earth is round."
        ):
            return (
                fast_analysis(
                    claim,
                    "supports",
                    "https://a.edu/earth",
                )
            )

        return None

    def fake_recovery(
        claim,
        existing_evidence,
        end_date=None,
        deadline=None,
    ):
        calls[
            "recovery"
        ] += 1

        return []

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
        .try_compound_fast_evidence_analysis(
            claim=(
                "Earth is round and "
                "Mars is red."
            ),
            subclaims=[
                "Earth is round.",
                "Mars is red.",
            ],
            evidence=[
                evidence(
                    "https://a.edu/earth"
                )
            ],
        )
    )

    assert (
        result[
            "analysis"
        ]
        is None
    )

    assert (
        calls[
            "recovery"
        ]
        == 1
    )


def test_recovery_can_resolve_subclaim(
    monkeypatch,
):
    recovered_url = (
        "https://b.edu/mars"
    )

    def fake_fast(
        claim,
        evidence,
    ):
        urls = {
            item.url
            for item in evidence
        }

        if (
            claim
            == "Earth is round."
        ):
            return (
                fast_analysis(
                    claim,
                    "supports",
                    "https://a.edu/earth",
                )
            )

        if (
            recovered_url
            in urls
        ):
            return (
                fast_analysis(
                    claim,
                    "supports",
                    recovered_url,
                )
            )

        return None

    monkeypatch.setattr(
        compound_fast_path,
        "try_fast_evidence_analysis",
        fake_fast,
    )

    monkeypatch.setattr(
        compound_fast_path,
        "recover_authority_evidence",
        lambda claim,
        existing_evidence,
        end_date=None,
        deadline=None: [
            evidence(
                recovered_url
            )
        ],
    )

    result = (
        compound_fast_path
        .try_compound_fast_evidence_analysis(
            claim=(
                "Earth is round and "
                "Mars is red."
            ),
            subclaims=[
                "Earth is round.",
                "Mars is red.",
            ],
            evidence=[
                evidence(
                    "https://a.edu/earth"
                )
            ],
        )
    )

    assert (
        result[
            "analysis"
        ]
        is not None
    )

    assert (
        result[
            "analysis"
        ][
            "overall_evidence_direction"
        ]
        == "supports"
    )

    assert (
        len(
            result[
                "recovered_evidence"
            ]
        )
        == 1
    )


def test_non_atomic_subclaim_is_rejected(
    monkeypatch,
):
    monkeypatch.setattr(
        compound_fast_path,
        "is_simple_atomic_claim",
        lambda claim: (
            " because "
            not in (
                " "
                + claim.lower()
                + " "
            )
        ),
    )

    result = (
        compound_fast_path
        .try_compound_fast_evidence_analysis(
            claim="Compound claim",
            subclaims=[
                (
                    "Earth orbits the Sun "
                    "because of gravity."
                ),
                "Mars has two moons.",
            ],
            evidence=[],
        )
    )

    assert (
        result[
            "analysis"
        ]
        is None
    )
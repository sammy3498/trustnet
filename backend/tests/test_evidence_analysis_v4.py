from backend.app.llm.evidence_analysis import (
    apply_v4_subclaim_consistency,
)


def build_analysis(
    *,
    coverage: str,
    direct_support: bool,
    direct_contradiction: bool,
    context_conflict: bool,
    status: str = "supports",
) -> dict:
    return {
        "subclaim_assessments": [
            {
                "subclaim": (
                    "Example factual "
                    "subclaim."
                ),
                "coverage": coverage,
                "direct_support": (
                    direct_support
                ),
                "direct_contradiction": (
                    direct_contradiction
                ),
                "material_context_conflict": (
                    context_conflict
                ),
                "status": status,
                "reasoning": "test",
            }
        ]
    }


def test_complete_direct_support():
    analysis = build_analysis(
        coverage="complete",
        direct_support=True,
        direct_contradiction=False,
        context_conflict=False,
        status="supports",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "supports"
    )


def test_partial_support_becomes_insufficient():
    analysis = build_analysis(
        coverage="partial",
        direct_support=True,
        direct_contradiction=False,
        context_conflict=False,
        status="supports",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "insufficient"
    )


def test_direct_contradiction_is_preserved():
    analysis = build_analysis(
        coverage="partial",
        direct_support=False,
        direct_contradiction=True,
        context_conflict=False,
        status="insufficient",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "contradicts"
    )


def test_support_and_contradiction_becomes_mixed():
    analysis = build_analysis(
        coverage="complete",
        direct_support=True,
        direct_contradiction=True,
        context_conflict=False,
        status="supports",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "mixed"
    )


def test_context_conflict_becomes_mixed():
    analysis = build_analysis(
        coverage="complete",
        direct_support=True,
        direct_contradiction=False,
        context_conflict=True,
        status="supports",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "mixed"
    )


def test_no_direct_evidence_becomes_insufficient():
    analysis = build_analysis(
        coverage="complete",
        direct_support=False,
        direct_contradiction=False,
        context_conflict=False,
        status="supports",
    )

    result = (
        apply_v4_subclaim_consistency(
            analysis
        )
    )

    subclaim = (
        result[
            "subclaim_assessments"
        ][0]
    )

    assert (
        subclaim["status"]
        == "insufficient"
    )
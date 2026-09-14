from backend.app.evidence.compound_fast_path import (
    combine_subclaim_analyses,
)
from backend.app.scoring.calibrator import (
    calibrate_factual_consistency,
)


def atomic_analysis(
    direction: str,
) -> dict:
    supports = (
        direction == "supports"
    )

    return {
        "evidence_assessments": [],
        "subclaim_assessments": [
            {
                "subclaim": "atomic",
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
        "overall_evidence_direction": direction,
        "material_caveats": [],
        "_fast_path": {
            "effective_evidence_count": 2,
            "effective_high_quality_count": 1,
            "effective_medium_quality_count": 1,
        },
    }


def test_compound_combiner_marks_and_operator():
    analysis = combine_subclaim_analyses(
        claim="A and B",
        subclaims=[
            "A",
            "B",
        ],
        analyses=[
            atomic_analysis(
                "supports"
            ),
            atomic_analysis(
                "contradicts"
            ),
        ],
    )

    assert analysis is not None
    assert (
        analysis[
            "_compound_operator"
        ]
        == "and"
    )
    assert (
        analysis[
            "overall_evidence_direction"
        ]
        == "contradicts"
    )


def test_and_support_plus_contradiction_scores_as_false():
    analysis = {
        "_compound_operator": "and",
        "subclaim_assessments": [
            {
                "status": "supports"
            },
            {
                "status": "contradicts"
            },
        ],
    }

    assert (
        calibrate_factual_consistency(
            analysis
        )
        == 5
    )


def test_generic_support_plus_contradiction_remains_mixed():
    analysis = {
        "subclaim_assessments": [
            {
                "status": "supports"
            },
            {
                "status": "contradicts"
            },
        ],
    }

    assert (
        calibrate_factual_consistency(
            analysis
        )
        == 50
    )


def test_and_all_supports_remains_high():
    analysis = {
        "_compound_operator": "and",
        "subclaim_assessments": [
            {
                "status": "supports"
            },
            {
                "status": "supports"
            },
        ],
    }

    assert (
        calibrate_factual_consistency(
            analysis
        )
        == 99
    )

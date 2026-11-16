from backend.app.evidence.direction import (
    align_evidence_strength_score,
    resolve_overall_evidence_direction,
)


def subclaim(
    status: str,
) -> dict:
    return {
        "subclaim": "test",
        "status": status,
        "reasoning": "test",
    }


def test_all_supported():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "supports"
                ),
                subclaim(
                    "supports"
                ),
            ],
            fallback_direction=(
                "insufficient"
            ),
        )
    )

    assert result == "supports"


def test_conjunction_with_false_component_is_contradicted():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "supports"
                ),
                subclaim(
                    "contradicts"
                ),
            ],
            fallback_direction=(
                "supports"
            ),
        )
    )

    assert result == "contradicts"


def test_mixed_subclaim_makes_overall_mixed():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "mixed"
                ),
                subclaim(
                    "supports"
                ),
            ],
            fallback_direction=(
                "supports"
            ),
        )
    )

    assert result == "mixed"


def test_supported_and_insufficient_is_insufficient():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "supports"
                ),
                subclaim(
                    "insufficient"
                ),
            ],
            fallback_direction=(
                "supports"
            ),
        )
    )

    assert result == "insufficient"


def test_contradiction_and_insufficient_is_contradiction():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "contradicts"
                ),
                subclaim(
                    "insufficient"
                ),
            ],
            fallback_direction=(
                "insufficient"
            ),
        )
    )

    assert result == "contradicts"


def test_all_insufficient():
    result = (
        resolve_overall_evidence_direction(
            [
                subclaim(
                    "insufficient"
                ),
                subclaim(
                    "insufficient"
                ),
            ],
            fallback_direction=(
                "supports"
            ),
        )
    )

    assert result == "insufficient"


def test_empty_subclaims_use_fallback():
    result = (
        resolve_overall_evidence_direction(
            [],
            fallback_direction=(
                "contradicts"
            ),
        )
    )

    assert result == "contradicts"


def test_changed_mixed_score_becomes_fifty():
    result = (
        align_evidence_strength_score(
            score=78,
            original_direction=(
                "supports"
            ),
            resolved_direction=(
                "mixed"
            ),
        )
    )

    assert result == 50


def test_changed_insufficient_score_becomes_fifty():
    result = (
        align_evidence_strength_score(
            score=82,
            original_direction=(
                "supports"
            ),
            resolved_direction=(
                "insufficient"
            ),
        )
    )

    assert result == 50


def test_unchanged_direction_keeps_score():
    result = (
        align_evidence_strength_score(
            score=82,
            original_direction=(
                "supports"
            ),
            resolved_direction=(
                "supports"
            ),
        )
    )

    assert result == 82
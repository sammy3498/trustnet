from backend.app.scoring.uncertainty import (
    evaluate_uncertainty,
)


def test_insufficient_evidence():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=1,
            high_quality_count=1,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="supports",
            medium_quality_count=0,
        )
    )

    assert gate is True
    assert level == "high"


def test_no_reliable_evidence():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=5,
            high_quality_count=0,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="supports",
            medium_quality_count=0,
        )
    )

    assert gate is True
    assert level == "high"


def test_one_medium_source_is_not_enough():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=3,
            high_quality_count=0,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="supports",
            medium_quality_count=1,
        )
    )

    assert gate is True
    assert level == "high"


def test_two_medium_sources_are_sufficient():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=3,
            high_quality_count=0,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="supports",
            medium_quality_count=2,
        )
    )

    assert gate is False
    assert level == "low"


def test_one_high_quality_source_is_sufficient():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=3,
            high_quality_count=1,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="supports",
            medium_quality_count=0,
        )
    )

    assert gate is False
    assert level == "low"


def test_reliable_conflicting_evidence_does_not_abstain():
    gate, level, reason = (
        evaluate_uncertainty(
            evidence_count=4,
            high_quality_count=1,
            conflicting_evidence=True,
            claim_is_clear=True,
            evidence_direction="mixed",
            medium_quality_count=2,
        )
    )

    assert gate is False
    assert level == "medium"
    assert "mixed" in reason.lower()


def test_mixed_direction_does_not_abstain():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=4,
            high_quality_count=1,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="mixed",
            medium_quality_count=1,
        )
    )

    assert gate is False
    assert level == "medium"


def test_unreliable_mixed_evidence_still_abstains():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=4,
            high_quality_count=0,
            conflicting_evidence=True,
            claim_is_clear=True,
            evidence_direction="mixed",
            medium_quality_count=1,
        )
    )

    assert gate is True
    assert level == "high"


def test_insufficient_direction_forces_gate():
    gate, level, _ = (
        evaluate_uncertainty(
            evidence_count=5,
            high_quality_count=3,
            conflicting_evidence=False,
            claim_is_clear=True,
            evidence_direction="insufficient",
            medium_quality_count=2,
        )
    )

    assert gate is True
    assert level == "high"
from backend.app.scoring.uncertainty import evaluate_uncertainty


def test_insufficient_evidence():
    result = evaluate_uncertainty(
        evidence_count=1,
        high_quality_count=1,
        conflicting_evidence=False,
        claim_is_clear=True,
    )

    assert result[0] is True
    assert result[1] == "high"


def test_no_high_quality_evidence():
    result = evaluate_uncertainty(
        evidence_count=3,
        high_quality_count=0,
        conflicting_evidence=False,
        claim_is_clear=True,
    )

    assert result[0] is True


def test_conflicting_evidence():
    result = evaluate_uncertainty(
        evidence_count=4,
        high_quality_count=2,
        conflicting_evidence=True,
        claim_is_clear=True,
    )

    assert result[0] is True
    assert result[1] == "medium"


def test_sufficient_evidence():
    result = evaluate_uncertainty(
        evidence_count=4,
        high_quality_count=2,
        conflicting_evidence=False,
        claim_is_clear=True,
    )

    assert result[0] is False
    assert result[1] == "low"
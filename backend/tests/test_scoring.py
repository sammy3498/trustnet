from backend.app.scoring.calculator import calculate_overall_score, get_label


def test_calculate_overall_score():
    score = calculate_overall_score(
        evidence_strength=90,
        factual_consistency=85,
        source_provenance=80,
        bias_framing=75,
        incentive_conflict=70,
    )

    assert score == 85


def test_highly_credible_label():
    assert get_label(85) == "Highly Credible"


def test_moderately_credible_label():
    assert get_label(70) == "Moderately Credible"


def test_low_credibility_label():
    assert get_label(50) == "Low Credibility"


def test_not_credible_label():
    assert get_label(20) == "Not Credible"


def test_uncertainty_override():
    assert get_label(90, True) == "Insufficient Evidence"
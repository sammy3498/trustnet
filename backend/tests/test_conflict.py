from backend.app.evidence.conflict import detect_conflicting_evidence


def test_support_only():
    evidence = [
        {"stance": "supports"},
        {"stance": "supports"},
    ]

    assert detect_conflicting_evidence(evidence) is False


def test_contradict_only():
    evidence = [
        {"stance": "contradicts"},
        {"stance": "contradicts"},
    ]

    assert detect_conflicting_evidence(evidence) is False


def test_support_and_contradict():
    evidence = [
        {"stance": "supports"},
        {"stance": "contradicts"},
    ]

    assert detect_conflicting_evidence(evidence) is True


def test_neutral_does_not_create_conflict():
    evidence = [
        {"stance": "supports"},
        {"stance": "neutral"},
    ]

    assert detect_conflicting_evidence(evidence) is False
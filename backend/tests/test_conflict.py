from backend.app.evidence.conflict import (
    detect_conflicting_evidence,
)


def test_support_only():
    evidence = [
        {
            "url": "https://a.com",
            "stance": "supports",
        },
        {
            "url": "https://b.com",
            "stance": "supports",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence
        )
        is False
    )


def test_contradict_only():
    evidence = [
        {
            "url": "https://a.com",
            "stance": "contradicts",
        },
        {
            "url": "https://b.com",
            "stance": "contradicts",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence
        )
        is False
    )


def test_support_and_contradict():
    evidence = [
        {
            "url": "https://a.com",
            "stance": "supports",
        },
        {
            "url": "https://b.com",
            "stance": "contradicts",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence
        )
        is True
    )


def test_neutral_does_not_create_conflict():
    evidence = [
        {
            "url": "https://a.com",
            "stance": "supports",
        },
        {
            "url": "https://b.com",
            "stance": "neutral",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence
        )
        is False
    )


def test_low_quality_support_does_not_create_reliable_conflict():
    evidence = [
        {
            "url": "https://reliable.com",
            "stance": "contradicts",
        },
        {
            "url": "https://low.com",
            "stance": "supports",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence,
            reliable_urls={
                "https://reliable.com",
            },
        )
        is False
    )


def test_low_quality_contradiction_does_not_create_reliable_conflict():
    evidence = [
        {
            "url": "https://reliable.com",
            "stance": "supports",
        },
        {
            "url": "https://low.com",
            "stance": "contradicts",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence,
            reliable_urls={
                "https://reliable.com",
            },
        )
        is False
    )


def test_reliable_support_and_contradiction_create_conflict():
    evidence = [
        {
            "url": "https://a.com",
            "stance": "supports",
        },
        {
            "url": "https://b.com",
            "stance": "contradicts",
        },
        {
            "url": "https://low.com",
            "stance": "neutral",
        },
    ]

    assert (
        detect_conflicting_evidence(
            evidence,
            reliable_urls={
                "https://a.com",
                "https://b.com",
            },
        )
        is True
    )
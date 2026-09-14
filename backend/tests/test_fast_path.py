from backend.app.evidence.fast_path import (
    build_fast_path_analysis,
    has_enough_term_overlap,
    is_simple_atomic_claim,
    score_evidence_sentences,
    sentences_are_duplicates,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)


def evidence_item(
    source: str,
    quality: str,
    text: str,
):
    return RetrievedEvidence(
        title="Test evidence",
        url=(
            f"https://{source}/test"
        ),
        source=source,
        text=text,
        quality=quality,
    )


def match(
    index: int,
    sentence: str,
    stance: str = "supports",
    confidence: float = 0.99,
):
    if stance == "supports":
        entailment = confidence
        contradiction = 0.001

    elif stance == "contradicts":
        entailment = 0.001
        contradiction = confidence

    else:
        entailment = 0.05
        contradiction = 0.05

    return {
        "evidence_index": index,
        "sentence": sentence,
        "stance": stance,
        "confidence": confidence,
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": max(
            0.0,
            1.0
            - entailment
            - contradiction,
        ),
    }


def test_simple_atomic_claim_is_eligible():
    assert is_simple_atomic_claim(
        "The Earth orbits the Sun."
    )


def test_compound_claim_is_not_eligible():
    assert not is_simple_atomic_claim(
        (
            "The Earth orbits the Sun "
            "and Mars has two moons."
        )
    )


def test_numeric_claim_is_not_eligible():
    assert not is_simple_atomic_claim(
        "Water boils at 100 degrees."
    )


def test_duplicate_sentence_detection():
    first = (
        "The amount of sunlight received "
        "on Earth's surface is affected "
        "by variations of Earth's orbit "
        "around the Sun."
    )

    second = (
        "The amount of sunlight received "
        "on Earth's surface is affected "
        "by variations of the Earth's "
        "orbit around the Sun."
    )

    assert sentences_are_duplicates(
        first,
        second,
    )


def test_relation_sentence_has_required_overlap():
    assert has_enough_term_overlap(
        "The Sun orbits the Earth.",
        (
            "The Earth follows an orbit "
            "around the Sun."
        ),
    )


def test_generic_entity_sentence_is_not_relation_evidence():
    assert not has_enough_term_overlap(
        "The Sun orbits the Earth.",
        (
            "The Sun warms the planet "
            "and makes life on Earth possible."
        ),
    )


def test_fast_path_accepts_independent_reliable_consensus():
    evidence = [
        evidence_item(
            "example.edu",
            "medium",
            "Evidence one.",
        ),
        evidence_item(
            "example.org",
            "medium",
            "Evidence two.",
        ),
    ]

    matches = [
        match(
            0,
            (
                "Earth travels around "
                "the Sun in an orbit."
            ),
        ),
        match(
            1,
            (
                "The Earth completes "
                "an orbit around the Sun."
            ),
        ),
    ]

    result = build_fast_path_analysis(
        claim=(
            "The Earth orbits the Sun."
        ),
        evidence=evidence,
        matches=matches,
    )

    assert result is not None

    assert (
        result[
            "overall_evidence_direction"
        ]
        == "supports"
    )

    assert (
        result[
            "_fast_path"
        ][
            "effective_evidence_count"
        ]
        == 2
    )


def test_duplicate_evidence_is_not_double_counted():
    evidence = [
        evidence_item(
            "first.edu",
            "medium",
            "Evidence one.",
        ),
        evidence_item(
            "second.edu",
            "medium",
            "Evidence two.",
        ),
        evidence_item(
            "third.edu",
            "medium",
            "Evidence three.",
        ),
    ]

    matches = [
        match(
            0,
            (
                "The Earth follows an orbit "
                "around the Sun each year."
            ),
        ),
        match(
            1,
            (
                "The Earth follows an orbit "
                "around the Sun each year."
            ),
        ),
        match(
            2,
            (
                "Earth moves around "
                "the Sun in its orbit."
            ),
        ),
    ]

    result = build_fast_path_analysis(
        claim=(
            "The Earth orbits the Sun."
        ),
        evidence=evidence,
        matches=matches,
    )

    assert result is not None

    assert (
        result[
            "_fast_path"
        ][
            "effective_evidence_count"
        ]
        == 2
    )


def test_low_quality_source_cannot_complete_fast_path():
    evidence = [
        evidence_item(
            "reliable.edu",
            "medium",
            "Reliable evidence.",
        ),
        evidence_item(
            "weak.example",
            "low",
            "Weak evidence.",
        ),
    ]

    matches = [
        match(
            0,
            "Earth travels around the Sun.",
        ),
        match(
            1,
            (
                "The Earth completes "
                "an orbit around the Sun."
            ),
        ),
    ]

    result = build_fast_path_analysis(
        claim=(
            "The Earth orbits the Sun."
        ),
        evidence=evidence,
        matches=matches,
    )

    assert result is None


def test_conflicting_independent_evidence_escalates():
    evidence = [
        evidence_item(
            "first.edu",
            "medium",
            "Evidence one.",
        ),
        evidence_item(
            "second.edu",
            "medium",
            "Evidence two.",
        ),
    ]

    matches = [
        match(
            0,
            "Earth travels around the Sun.",
            stance="supports",
        ),
        match(
            1,
            (
                "Earth does not travel "
                "around the Sun."
            ),
            stance="contradicts",
        ),
    ]

    result = build_fast_path_analysis(
        claim=(
            "The Earth orbits the Sun."
        ),
        evidence=evidence,
        matches=matches,
    )

    assert result is None


def test_nli_model_handles_relation_reversal():
    evidence = [
        evidence_item(
            "science.example.edu",
            "medium",
            (
                "The Earth completes an orbit "
                "around the Sun."
            ),
        )
    ]

    true_matches = (
        score_evidence_sentences(
            claim=(
                "The Earth orbits "
                "the Sun."
            ),
            evidence=evidence,
        )
    )

    assert len(
        true_matches
    ) == 1

    assert (
        true_matches[
            0
        ][
            "stance"
        ]
        == "supports"
    )

    assert (
        true_matches[
            0
        ][
            "entailment"
        ]
        >= 0.95
    )

    false_matches = (
        score_evidence_sentences(
            claim=(
                "The Sun orbits "
                "the Earth."
            ),
            evidence=evidence,
        )
    )

    assert len(
        false_matches
    ) == 1

    assert (
        false_matches[
            0
        ][
            "stance"
        ]
        == "contradicts"
    )

    assert (
        false_matches[
            0
        ][
            "contradiction"
        ]
        >= 0.95
    )
from backend.app.evidence import fast_path
from backend.app.evidence.models import (
    RetrievedEvidence,
)


def evidence(
    url: str,
    source: str,
    text: str,
    quality: str,
) -> RetrievedEvidence:
    return RetrievedEvidence(
        title=source,
        url=url,
        source=source,
        text=text,
        quality=quality,
        evidence_origin="webpage",
    )


def test_cardinality_one_of_is_non_exhaustive():
    assert (
        fast_path
        .has_non_exhaustive_cardinality_phrase(
            claim="Mars has one moon.",
            sentence=(
                "Deimos is one of the "
                "smallest known moons of Mars."
            ),
        )
    )


def test_cardinality_one_of_is_filtered_before_nli(
    monkeypatch,
):
    items = [
        evidence(
            url="https://example-a.gov/deimos",
            source="example-a.gov",
            text=(
                "Deimos is one of the "
                "smallest known moons of Mars."
            ),
            quality="high",
        ),
        evidence(
            url="https://example-b.edu/phobos",
            source="example-b.edu",
            text=(
                "Phobos is one of the "
                "natural moons of Mars."
            ),
            quality="medium",
        ),
    ]

    def fail_if_model_is_loaded():
        raise AssertionError(
            "NLI should not run for "
            "non-exhaustive cardinality "
            "sentences."
        )

    monkeypatch.setattr(
        fast_path,
        "get_fast_path_model",
        fail_if_model_is_loaded,
    )

    matches = (
        fast_path
        .score_evidence_sentences(
            claim="Mars has one moon.",
            evidence=items,
        )
    )

    assert matches == []


def test_mars_one_moon_is_contradicted_by_two_sources():
    items = [
        evidence(
            url=(
                "https://science.nasa.gov/"
                "mars/moons/"
            ),
            source="science.nasa.gov",
            text=(
                "Mars has two small moons, "
                "Phobos and Deimos."
            ),
            quality="high",
        ),
        evidence(
            url=(
                "https://sos.noaa.gov/"
                "catalog/datasets/mars/"
            ),
            source="sos.noaa.gov",
            text=(
                "The two moons of Mars are "
                "Phobos and Deimos."
            ),
            quality="high",
        ),
    ]

    analysis = (
        fast_path
        .try_fast_evidence_analysis(
            claim="Mars has one moon.",
            evidence=items,
        )
    )

    assert analysis is not None

    assert (
        analysis[
            "overall_evidence_direction"
        ]
        == "contradicts"
    )


def test_mars_two_moons_is_supported_by_two_sources():
    items = [
        evidence(
            url=(
                "https://science.nasa.gov/"
                "mars/moons/"
            ),
            source="science.nasa.gov",
            text=(
                "Mars has two small moons, "
                "Phobos and Deimos."
            ),
            quality="high",
        ),
        evidence(
            url=(
                "https://sos.noaa.gov/"
                "catalog/datasets/mars/"
            ),
            source="sos.noaa.gov",
            text=(
                "The two moons of Mars are "
                "Phobos and Deimos."
            ),
            quality="high",
        ),
    ]

    analysis = (
        fast_path
        .try_fast_evidence_analysis(
            claim="Mars has two moons.",
            evidence=items,
        )
    )

    assert analysis is not None

    assert (
        analysis[
            "overall_evidence_direction"
        ]
        == "supports"
    )

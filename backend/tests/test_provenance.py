from backend.app.evidence.provenance import (
    classify_source_quality,
)


def test_government_domain_is_high():
    assert (
        classify_source_quality(
            "https://example.gov/report"
        )
        == "high"
    )


def test_nih_subdomain_is_high():
    assert (
        classify_source_quality(
            "https://pmc.ncbi.nlm.nih.gov/article"
        )
        == "high"
    )


def test_reuters_is_high():
    assert (
        classify_source_quality(
            "https://www.reuters.com/article"
        )
        == "high"
    )


def test_university_domain_is_medium():
    assert (
        classify_source_quality(
            "https://news.climate.columbia.edu/article"
        )
        == "medium"
    )


def test_statnews_is_medium():
    assert (
        classify_source_quality(
            "https://www.statnews.com/article"
        )
        == "medium"
    )


def test_bbc_is_medium():
    assert (
        classify_source_quality(
            "https://www.bbc.com/news/article"
        )
        == "medium"
    )


def test_snopes_is_medium():
    assert (
        classify_source_quality(
            "https://www.snopes.com/fact-check/example"
        )
        == "medium"
    )


def test_factcheck_is_medium():
    assert (
        classify_source_quality(
            "https://www.factcheck.org/example"
        )
        == "medium"
    )


def test_fullfact_is_medium():
    assert (
        classify_source_quality(
            "https://fullfact.org/example"
        )
        == "medium"
    )


def test_cnet_is_medium():
    assert (
        classify_source_quality(
            "https://www.cnet.com/tech/example"
        )
        == "medium"
    )


def test_known_subdomain_is_medium():
    assert (
        classify_source_quality(
            "https://news.cnet.com/example"
        )
        == "medium"
    )


def test_unknown_domain_is_low():
    assert (
        classify_source_quality(
            "https://random-example-site.com/article"
        )
        == "low"
    )
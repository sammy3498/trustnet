from backend.app.retrieval import (
    evidence_builder,
)
from backend.app.retrieval.web_fetcher import (
    FetchedPageText,
    extract_publication_date_from_html,
    normalize_publication_date,
)


def test_article_published_time():
    html = (
        '<meta property="article:published_time" '
        'content="2021-04-08T10:30:00Z">'
    )

    assert (
        extract_publication_date_from_html(
            html
        )
        == "2021-04-08"
    )


def test_citation_publication_date():
    html = (
        '<meta name="citation_publication_date" '
        'content="April 28, 2022">'
    )

    assert (
        extract_publication_date_from_html(
            html
        )
        == "2022-04-28"
    )


def test_json_ld_date_published():
    html = """
    <script type="application/ld+json">
    {
        "@type": "NewsArticle",
        "datePublished": "2020-10-22T08:00:00Z"
    }
    </script>
    """

    assert (
        extract_publication_date_from_html(
            html
        )
        == "2020-10-22"
    )


def test_release_date_html_field():
    html = """
    <div
        class="field field--name-field-release-date field--type-string"
    >
        EMBARGOED UNTIL RELEASE AT 8:30 A.M. EST,
        Wednesday, November 25, 2020
    </div>
    """

    assert (
        extract_publication_date_from_html(
            html
        )
        == "2020-11-25"
    )


def test_embedded_written_date():
    value = (
        "EMBARGOED UNTIL RELEASE AT "
        "8:30 A.M. EST, Wednesday, "
        "November 25, 2020"
    )

    assert (
        normalize_publication_date(
            value
        )
        == "2020-11-25"
    )


def test_no_publication_date():
    html = (
        "<html><body>"
        "No metadata"
        "</body></html>"
    )

    assert (
        extract_publication_date_from_html(
            html
        )
        is None
    )


def test_normalize_written_date():
    assert (
        normalize_publication_date(
            "8 April 2021"
        )
        == "2021-04-08"
    )


def test_evidence_builder_preserves_date(
    monkeypatch,
):
    fetched = FetchedPageText(
        "Article body",
        published_date="2021-04-08",
    )

    monkeypatch.setattr(
        evidence_builder,
        "fetch_webpage",
        lambda url: fetched,
    )

    result = (
        evidence_builder.build_evidence_from_url(
            "https://example.com/article",
            "Example",
        )
    )

    assert (
        result.published_date
        == "2021-04-08"
    )
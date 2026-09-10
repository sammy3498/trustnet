from urllib.parse import urlparse

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.evidence.provenance import (
    classify_source_quality,
)
from backend.app.retrieval.web_fetcher import (
    fetch_webpage,
)


def get_source_domain(
    url: str,
) -> str:
    domain = (
        urlparse(
            url
        ).netloc.lower()
    )

    if domain.startswith(
        "www."
    ):
        domain = domain[
            4:
        ]

    return domain


def build_evidence_from_url(
    url: str,
    title: str = "",
) -> RetrievedEvidence:
    fetched_text = (
        fetch_webpage(
            url
        )
    )

    published_date = getattr(
        fetched_text,
        "published_date",
        None,
    )

    domain = (
        get_source_domain(
            url
        )
    )

    quality = (
        classify_source_quality(
            url
        )
    )

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=str(
            fetched_text
        ),
        quality=quality,
        evidence_origin="webpage",
        published_date=published_date,
    )


def build_evidence_from_snippet(
    url: str,
    snippet: str,
    title: str = "",
) -> RetrievedEvidence:
    domain = (
        get_source_domain(
            url
        )
    )

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=snippet,
        quality="low",
        evidence_origin="snippet",
        published_date=None,
    )
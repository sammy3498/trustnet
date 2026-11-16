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


def is_access_wall(text: str) -> bool:
    sample = " ".join(text.lower().split())
    if not sample:
        return True
    if len(sample) > 900:
        return False
    return any(phrase in sample for phrase in (
        "please enable javascript", "javascript is disabled",
        "sign in to continue", "log in to continue",
        "log in or sign up", "sign up to continue",
        "enable cookies to continue", "please accept cookies",
        "this site uses cookies", "this website uses cookies",
        "only displays a cookie", "please log in",
    ))


def build_evidence_from_url(
    url: str,
    title: str = "",
) -> RetrievedEvidence:
    fetched_text = (
        fetch_webpage(
            url
        )
    )

    # A successful HTTP response is not always a readable source: social
    # login screens, consent-only pages and JavaScript placeholders must not
    # be treated as direct evidence.  Raising here activates the existing
    # search-snippet fallback, with appropriately reduced confidence.
    if is_access_wall(str(fetched_text)):
        raise ValueError("Retrieved page contained an access or cookie wall rather than article evidence")

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

    quality = (
        classify_source_quality(
            url
        )
    )

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=snippet,
        quality=quality,
        evidence_origin="snippet",
        published_date=None,
    )

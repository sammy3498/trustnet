from urllib.parse import urlparse

from backend.app.evidence.models import RetrievedEvidence
from backend.app.evidence.provenance import classify_source_quality
from backend.app.retrieval.web_fetcher import fetch_webpage


def get_source_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def build_evidence_from_url(
    url: str,
    title: str = "",
) -> RetrievedEvidence:
    text = fetch_webpage(url)

    domain = get_source_domain(url)
    quality = classify_source_quality(url)

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=text,
        quality=quality,
        evidence_origin="webpage",
    )


def build_evidence_from_snippet(
    url: str,
    snippet: str,
    title: str = "",
) -> RetrievedEvidence:
    domain = get_source_domain(url)

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=snippet,
        quality="low",
        evidence_origin="snippet",
    )
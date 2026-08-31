from urllib.parse import urlparse

from backend.app.evidence.models import RetrievedEvidence
from backend.app.evidence.provenance import classify_source_quality
from backend.app.retrieval.web_fetcher import fetch_webpage


def build_evidence_from_url(
    url: str,
    title: str = "",
) -> RetrievedEvidence:
    text = fetch_webpage(url)

    domain = urlparse(url).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    quality = classify_source_quality(url)

    return RetrievedEvidence(
        title=title or domain,
        url=url,
        source=domain,
        text=text,
        quality=quality,
    )
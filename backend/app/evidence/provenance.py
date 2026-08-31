from urllib.parse import urlparse


HIGH_QUALITY_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "reuters.com",
    "apnews.com",
    "gov.uk",
    "iana.org",
    "nasa.gov",
}


MEDIUM_QUALITY_DOMAINS = {
    "bbc.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
}


def get_domain(url: str) -> str:
    domain = urlparse(url).netloc.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def domain_matches(domain: str, known_domain: str) -> bool:
    return (
        domain == known_domain
        or domain.endswith("." + known_domain)
    )


def classify_source_quality(url: str) -> str:
    domain = get_domain(url)

    for known_domain in HIGH_QUALITY_DOMAINS:
        if domain_matches(domain, known_domain):
            return "high"

    for known_domain in MEDIUM_QUALITY_DOMAINS:
        if domain_matches(domain, known_domain):
            return "medium"

    return "low"
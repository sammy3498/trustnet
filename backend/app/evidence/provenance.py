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


def classify_source_quality(url: str) -> str:
    domain = get_domain(url)

    if domain in HIGH_QUALITY_DOMAINS:
        return "high"

    if domain in MEDIUM_QUALITY_DOMAINS:
        return "medium"

    return "low"
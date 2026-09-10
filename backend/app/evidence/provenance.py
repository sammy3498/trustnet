from urllib.parse import urlparse


HIGH_QUALITY_DOMAINS = {
    "who.int",
    "nih.gov",
    "pubmed.ncbi.nlm.nih.gov",
    "nature.com",
    "science.org",
    "nejm.org",
    "jamanetwork.com",
    "thelancet.com",
    "plos.org",
    "bmj.com",
    "cochranelibrary.com",
    "reuters.com",
    "apnews.com",
    "gov.uk",
    "nasa.gov",
    "iana.org",
    "europa.eu",
    "un.org",
    "oecd.org",
    "worldbank.org",
}


ESTABLISHED_NEWS_DOMAINS = {
    "bbc.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "nbcnews.com",
    "abcnews.com",
    "cbsnews.com",
    "npr.org",
    "statnews.com",
    "cnet.com",
}


FACT_CHECKING_DOMAINS = {
    "factcheck.org",
    "politifact.com",
    "snopes.com",
    "fullfact.org",
}


RESEARCH_INFORMATION_DOMAINS = {
    "ourworldindata.org",
    "arxiv.org",
    "medrxiv.org",
}


MEDIUM_QUALITY_DOMAINS = (
    ESTABLISHED_NEWS_DOMAINS
    | FACT_CHECKING_DOMAINS
    | RESEARCH_INFORMATION_DOMAINS
)


HIGH_QUALITY_SUFFIXES = {
    ".gov",
    ".mil",
}


MEDIUM_QUALITY_SUFFIXES = {
    ".edu",
    ".ac.uk",
    ".edu.au",
}


def get_domain(
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


def domain_matches(
    domain: str,
    known_domain: str,
) -> bool:
    return (
        domain == known_domain
        or domain.endswith(
            "." + known_domain
        )
    )


def domain_has_suffix(
    domain: str,
    suffix: str,
) -> bool:
    return domain.endswith(
        suffix
    )


def classify_source_quality(
    url: str,
) -> str:
    domain = get_domain(
        url
    )

    for known_domain in (
        HIGH_QUALITY_DOMAINS
    ):
        if domain_matches(
            domain,
            known_domain,
        ):
            return "high"

    for suffix in (
        HIGH_QUALITY_SUFFIXES
    ):
        if domain_has_suffix(
            domain,
            suffix,
        ):
            return "high"

    for known_domain in (
        MEDIUM_QUALITY_DOMAINS
    ):
        if domain_matches(
            domain,
            known_domain,
        ):
            return "medium"

    for suffix in (
        MEDIUM_QUALITY_SUFFIXES
    ):
        if domain_has_suffix(
            domain,
            suffix,
        ):
            return "medium"

    return "low"
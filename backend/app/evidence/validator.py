from urllib.parse import urlparse


def is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def validate_evidence(
    title: str,
    url: str,
    summary: str,
    source: str,
) -> tuple[bool, list[str]]:
    errors = []

    if not title.strip():
        errors.append("Missing evidence title.")

    if not is_valid_url(url):
        errors.append("Invalid evidence URL.")

    if not summary.strip():
        errors.append("Missing evidence summary.")

    if not source.strip():
        errors.append("Missing evidence source.")

    return len(errors) == 0, errors
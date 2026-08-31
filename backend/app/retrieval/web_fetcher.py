import httpx
import trafilatura


def fetch_webpage(url: str) -> str:
    headers = {
        "User-Agent": "TrustNet/0.1"
    }

    with httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers=headers,
    ) as client:
        response = client.get(url)
        response.raise_for_status()

    text = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
    )

    if not text:
        raise ValueError(
            f"Could not extract readable content from {url}"
        )

    return text
from urllib.parse import urlparse

import tldextract


extractor = tldextract.TLDExtract(
    suffix_list_urls=(),
)


def get_publisher_domain(value: str) -> str:
    if "://" in value:
        host = urlparse(value).netloc.lower()
    else:
        host = value.lower()

    if host.startswith("www."):
        host = host[4:]

    extracted = extractor(host)

    if not extracted.domain or not extracted.suffix:
        return host

    return f"{extracted.domain}.{extracted.suffix}"
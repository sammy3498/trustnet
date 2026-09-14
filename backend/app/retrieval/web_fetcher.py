import atexit
import json
import re
from datetime import datetime
from email.utils import (
    parsedate_to_datetime,
)
from html.parser import HTMLParser
from io import BytesIO

import httpx
import trafilatura
from pypdf import PdfReader


PUBLICATION_META_KEYS = {
    "article:published_time",
    "og:published_time",
    "citation_publication_date",
    "citation_date",
    "datepublished",
    "date_published",
    "publication_date",
    "publishdate",
    "publish_date",
    "pubdate",
    "dc.date",
    "dc.date.issued",
    "dcterms.date",
    "dcterms.issued",
    "prism.publicationdate",
}

PUBLICATION_FIELD_HINTS = {
    "release-date",
    "release_date",
    "publication-date",
    "publication_date",
    "published-date",
    "published_date",
    "publish-date",
    "publish_date",
    "date-published",
    "date_published",
    "datepublished",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}

MONTH_PATTERN = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

MAX_PDF_PAGES = 80
MAX_PDF_CHARS = 250000

HTTP_TIMEOUT_SECONDS = 3.0

_http_client = httpx.Client(
    timeout=HTTP_TIMEOUT_SECONDS,
    follow_redirects=True,
    headers={
        "User-Agent": "TrustNet/0.1"
    },
    limits=httpx.Limits(
        max_connections=20,
        max_keepalive_connections=10,
    ),
)


def close_http_client() -> None:
    _http_client.close()


atexit.register(
    close_http_client
)


class FetchedPageText(str):
    def __new__(
        cls,
        value: str,
        published_date: str | None = None,
    ):
        obj = super().__new__(
            cls,
            value,
        )

        obj.published_date = (
            published_date
        )

        return obj


def looks_like_publication_field(
    attributes: dict[str, str],
) -> bool:
    values = []

    for key in (
        "class",
        "id",
        "name",
        "property",
        "itemprop",
    ):
        value = attributes.get(
            key
        )

        if value:
            values.append(
                value.lower()
            )

    combined = " ".join(
        values
    )

    return any(
        hint in combined
        for hint in PUBLICATION_FIELD_HINTS
    )


class PublicationMetadataParser(
    HTMLParser
):
    def __init__(self):
        super().__init__()

        self.candidates = []

        self.in_json_ld = False
        self.json_ld_parts = []

        self.capture_depth = 0
        self.capture_parts = []

    def handle_starttag(
        self,
        tag,
        attrs,
    ):
        tag = tag.lower()

        attributes = {
            key.lower(): value
            for key, value in attrs
            if key and value is not None
        }

        if self.capture_depth > 0:
            if tag not in VOID_TAGS:
                self.capture_depth += 1

            return

        if tag == "meta":
            key = (
                attributes.get(
                    "property"
                )
                or attributes.get(
                    "name"
                )
                or attributes.get(
                    "itemprop"
                )
            )

            content = (
                attributes.get(
                    "content"
                )
            )

            if (
                key
                and content
                and key.lower()
                in PUBLICATION_META_KEYS
            ):
                self.candidates.append(
                    content.strip()
                )

        if tag == "script":
            script_type = (
                attributes.get(
                    "type",
                    ""
                ).lower()
            )

            if (
                "ld+json"
                in script_type
            ):
                self.in_json_ld = True
                self.json_ld_parts = []

        if looks_like_publication_field(
            attributes
        ):
            for attribute_name in (
                "datetime",
                "content",
                "value",
            ):
                value = attributes.get(
                    attribute_name
                )

                if value:
                    self.candidates.append(
                        value.strip()
                    )

            if tag not in VOID_TAGS:
                self.capture_depth = 1
                self.capture_parts = []

    def handle_data(
        self,
        data,
    ):
        if self.in_json_ld:
            self.json_ld_parts.append(
                data
            )

        if self.capture_depth > 0:
            self.capture_parts.append(
                data
            )

    def handle_endtag(
        self,
        tag,
    ):
        tag = tag.lower()

        if (
            tag == "script"
            and self.in_json_ld
        ):
            raw = "".join(
                self.json_ld_parts
            ).strip()

            self.in_json_ld = False
            self.json_ld_parts = []

            if raw:
                try:
                    data = json.loads(
                        raw
                    )

                    self._collect_json_ld_dates(
                        data
                    )

                except json.JSONDecodeError:
                    pass

        if self.capture_depth > 0:
            self.capture_depth -= 1

            if self.capture_depth == 0:
                text = " ".join(
                    part.strip()
                    for part
                    in self.capture_parts
                    if part.strip()
                )

                self.capture_parts = []

                if text:
                    self.candidates.append(
                        text
                    )

    def _collect_json_ld_dates(
        self,
        value,
    ):
        if isinstance(
            value,
            dict,
        ):
            for key, item in (
                value.items()
            ):
                if (
                    key.lower()
                    == "datepublished"
                    and isinstance(
                        item,
                        str,
                    )
                ):
                    self.candidates.append(
                        item.strip()
                    )

                self._collect_json_ld_dates(
                    item
                )

        elif isinstance(
            value,
            list,
        ):
            for item in value:
                self._collect_json_ld_dates(
                    item
                )


def month_number(
    month: str,
) -> int:
    month = month.strip(
        "."
    )

    if month.lower() == "sept":
        return 9

    for format_string in (
        "%B",
        "%b",
    ):
        try:
            return datetime.strptime(
                month,
                format_string,
            ).month

        except ValueError:
            continue

    raise ValueError(
        f"Invalid month: {month}"
    )


def normalize_publication_date(
    value: str,
) -> str | None:
    if not value:
        return None

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    iso_match = re.search(
        r"(?<!\d)"
        r"(19\d{2}|20\d{2})"
        r"-(\d{2})"
        r"-(\d{2})"
        r"(?!\d)",
        value,
    )

    if iso_match:
        try:
            parsed = datetime(
                int(
                    iso_match.group(1)
                ),
                int(
                    iso_match.group(2)
                ),
                int(
                    iso_match.group(3)
                ),
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            pass

    slash_match = re.search(
        r"(?<!\d)"
        r"(19\d{2}|20\d{2})"
        r"/(\d{1,2})"
        r"/(\d{1,2})"
        r"(?!\d)",
        value,
    )

    if slash_match:
        try:
            parsed = datetime(
                int(
                    slash_match.group(1)
                ),
                int(
                    slash_match.group(2)
                ),
                int(
                    slash_match.group(3)
                ),
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            pass

    month_first = re.search(
        rf"\b({MONTH_PATTERN})\.?\s+"
        r"(\d{1,2})"
        r"(?:st|nd|rd|th)?"
        r",?\s+"
        r"(19\d{2}|20\d{2})\b",
        value,
        re.IGNORECASE,
    )

    if month_first:
        try:
            parsed = datetime(
                int(
                    month_first.group(3)
                ),
                month_number(
                    month_first.group(1)
                ),
                int(
                    month_first.group(2)
                ),
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            pass

    day_first = re.search(
        r"\b(\d{1,2})"
        r"(?:st|nd|rd|th)?\s+"
        rf"({MONTH_PATTERN})\.?\s+"
        r"(19\d{2}|20\d{2})\b",
        value,
        re.IGNORECASE,
    )

    if day_first:
        try:
            parsed = datetime(
                int(
                    day_first.group(3)
                ),
                month_number(
                    day_first.group(2)
                ),
                int(
                    day_first.group(1)
                ),
            )

            return parsed.strftime(
                "%Y-%m-%d"
            )

        except ValueError:
            pass

    try:
        parsed = (
            parsedate_to_datetime(
                value
            )
        )

        return parsed.strftime(
            "%Y-%m-%d"
        )

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        return None


def extract_publication_date_from_html(
    html: str,
) -> str | None:
    parser = (
        PublicationMetadataParser()
    )

    try:
        parser.feed(
            html
        )

    except Exception:
        return None

    for candidate in (
        parser.candidates
    ):
        normalized = (
            normalize_publication_date(
                candidate
            )
        )

        if normalized:
            return normalized

    return None


def is_pdf_content(
    url: str,
    content_type: str,
) -> bool:
    clean_url = (
        url.split(
            "?",
            1,
        )[0]
        .split(
            "#",
            1,
        )[0]
        .lower()
    )

    return (
        "application/pdf"
        in content_type.lower()
        or clean_url.endswith(
            ".pdf"
        )
    )


def extract_pdf_text(
    content: bytes,
) -> str:
    reader = PdfReader(
        BytesIO(
            content
        )
    )

    parts = []
    total_chars = 0

    for page_number, page in enumerate(
        reader.pages
    ):
        if (
            page_number
            >= MAX_PDF_PAGES
        ):
            break

        text = (
            page.extract_text()
            or ""
        ).strip()

        if not text:
            continue

        remaining = (
            MAX_PDF_CHARS
            - total_chars
        )

        if remaining <= 0:
            break

        text = text[
            :remaining
        ]

        parts.append(
            text
        )

        total_chars += len(
            text
        )

    result = "\n\n".join(
        parts
    ).strip()

    if not result:
        raise ValueError(
            "Could not extract readable "
            "text from PDF."
        )

    return result


def fetch_webpage(
    url: str,
) -> str:
    response = _http_client.get(
        url
    )

    response.raise_for_status()

    content_type = (
        response.headers.get(
            "content-type",
            "",
        )
    )

    if is_pdf_content(
        url,
        content_type,
    ):
        text = extract_pdf_text(
            response.content
        )

        return FetchedPageText(
            text,
            published_date=None,
        )

    published_date = (
        extract_publication_date_from_html(
            response.text
        )
    )

    text = trafilatura.extract(
        response.text,
        include_comments=False,
        include_tables=True,
    )

    if not text:
        raise ValueError(
            "Could not extract readable "
            f"content from {url}"
        )

    return FetchedPageText(
        text,
        published_date=published_date,
    )
import re
from datetime import datetime


YEAR_PATTERN = re.compile(
    r"\b(19\d{2}|20\d{2})\b"
)

NUMERIC_DATE_PATTERN = re.compile(
    r"(19\d{2}|20\d{2})"
    r"[/_-]"
    r"(0?[1-9]|1[0-2])"
    r"[/_-]"
    r"(0?[1-9]|[12]\d|3[01])"
)

MONTH_NAME = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

MONTH_FIRST_PATTERN = re.compile(
    rf"\b({MONTH_NAME})\s+"
    r"(\d{1,2}),?\s+"
    r"(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

DAY_FIRST_PATTERN = re.compile(
    r"\b(\d{1,2})\s+"
    rf"({MONTH_NAME})\s+"
    r"(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)

TEMPORAL_LABEL_PATTERN = re.compile(
    r"\b("
    r"published|"
    r"publication date|"
    r"date published|"
    r"updated|"
    r"last updated|"
    r"modified"
    r")\b",
    re.IGNORECASE,
)

CURRENT_STATE_PATTERN = re.compile(
    r"\b("
    r"currently|"
    r"current|"
    r"latest|"
    r"today|"
    r"now|"
    r"most recent|"
    r"as of|"
    r"previous quarter|"
    r"previous month|"
    r"previous year|"
    r"last quarter|"
    r"last month|"
    r"year to date|"
    r"ytd"
    r")\b",
    re.IGNORECASE,
)


def extract_explicit_years(
    text: str,
) -> list[int]:
    if not text:
        return []

    return [
        int(match)
        for match in (
            YEAR_PATTERN.findall(
                text
            )
        )
    ]


def _month_number(
    month: str,
) -> int:
    for format_string in (
        "%B",
        "%b",
    ):
        try:
            return (
                datetime.strptime(
                    month,
                    format_string,
                ).month
            )

        except ValueError:
            continue

    if month.lower() == "sept":
        return 9

    raise ValueError(
        f"Invalid month: {month}"
    )


def extract_explicit_dates(
    text: str,
) -> list[datetime]:
    if not text:
        return []

    dates = []

    for year, month, day in (
        NUMERIC_DATE_PATTERN.findall(
            text
        )
    ):
        try:
            dates.append(
                datetime(
                    int(year),
                    int(month),
                    int(day),
                )
            )

        except ValueError:
            continue

    for month, day, year in (
        MONTH_FIRST_PATTERN.findall(
            text
        )
    ):
        try:
            dates.append(
                datetime(
                    int(year),
                    _month_number(
                        month
                    ),
                    int(day),
                )
            )

        except ValueError:
            continue

    for day, month, year in (
        DAY_FIRST_PATTERN.findall(
            text
        )
    ):
        try:
            dates.append(
                datetime(
                    int(year),
                    _month_number(
                        month
                    ),
                    int(day),
                )
            )

        except ValueError:
            continue

    return dates


def extract_labeled_temporal_dates(
    text: str,
    max_chars: int = 5000,
) -> list[datetime]:
    if not text:
        return []

    sample = text[
        :max_chars
    ]

    dates = []

    for match in (
        TEMPORAL_LABEL_PATTERN.finditer(
            sample
        )
    ):
        start = match.start()

        end = min(
            len(sample),
            match.end() + 100,
        )

        dates.extend(
            extract_explicit_dates(
                sample[
                    start:end
                ]
            )
        )

    return dates


def has_future_current_state_year(
    text: str,
    end_date: str | None,
    max_chars: int = 1500,
) -> bool:
    if (
        not text
        or not end_date
    ):
        return False

    try:
        cutoff = datetime.strptime(
            end_date,
            "%Y-%m-%d",
        )

    except ValueError:
        return False

    sample = text[
        :max_chars
    ]

    for match in (
        YEAR_PATTERN.finditer(
            sample
        )
    ):
        year = int(
            match.group(1)
        )

        if year <= cutoff.year:
            continue

        start = max(
            0,
            match.start() - 250,
        )

        end = min(
            len(sample),
            match.end() + 250,
        )

        context = sample[
            start:end
        ]

        if CURRENT_STATE_PATTERN.search(
            context
        ):
            return True

    return False


def is_explicitly_after_cutoff(
    title: str,
    url: str,
    end_date: str | None,
) -> bool:
    if not end_date:
        return False

    cutoff = datetime.strptime(
        end_date,
        "%Y-%m-%d",
    )

    explicit_dates = []

    explicit_dates.extend(
        extract_explicit_dates(
            title
        )
    )

    explicit_dates.extend(
        extract_explicit_dates(
            url
        )
    )

    if any(
        date > cutoff
        for date in explicit_dates
    ):
        return True

    years = []

    years.extend(
        extract_explicit_years(
            title
        )
    )

    years.extend(
        extract_explicit_years(
            url
        )
    )

    return any(
        year > cutoff.year
        for year in years
    )


def is_publication_date_after_cutoff(
    published_date: str | None,
    end_date: str | None,
) -> bool:
    if (
        not published_date
        or not end_date
    ):
        return False

    try:
        published = datetime.strptime(
            published_date,
            "%Y-%m-%d",
        )

        cutoff = datetime.strptime(
            end_date,
            "%Y-%m-%d",
        )

    except ValueError:
        return False

    return published > cutoff


def is_text_explicitly_after_cutoff(
    text: str,
    end_date: str | None,
) -> bool:
    if not end_date:
        return False

    cutoff = datetime.strptime(
        end_date,
        "%Y-%m-%d",
    )

    dates = (
        extract_labeled_temporal_dates(
            text
        )
    )

    if any(
        date > cutoff
        for date in dates
    ):
        return True

    return has_future_current_state_year(
        text,
        end_date,
    )
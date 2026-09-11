from backend.app.retrieval.temporal import (
    get_soft_temporal_fit_score,
)
from backend.app.retrieval.web_fetcher import (
    is_pdf_content,
)


def test_pdf_detected_from_url():
    assert is_pdf_content(
        "https://example.com/report.pdf",
        "application/octet-stream",
    )


def test_pdf_detected_from_content_type():
    assert is_pdf_content(
        "https://example.com/report",
        "application/pdf",
    )


def test_future_current_source_gets_low_temporal_fit():
    score = (
        get_soft_temporal_fit_score(
            reference_text=(
                "South Africa tourist "
                "arrivals 2018 2019"
            ),
            title=(
                "South Africa Tourist "
                "Arrivals"
            ),
            text=(
                "Current tourism statistics "
                "for 2026."
            ),
            published_date=(
                "2026-02-03"
            ),
        )
    )

    assert score < 0.5


def test_later_source_can_still_match_historical_year():
    score = (
        get_soft_temporal_fit_score(
            reference_text=(
                "South Africa tourist "
                "arrivals 2018 2019"
            ),
            title=(
                "Historical tourism data"
            ),
            text=(
                "In 2018 South Africa "
                "received 10 million "
                "international tourists."
            ),
            published_date=(
                "2026-02-03"
            ),
        )
    )

    assert score == 1.0
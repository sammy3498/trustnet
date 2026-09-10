from backend.app.retrieval.temporal import (
    extract_explicit_dates,
    extract_explicit_years,
    extract_labeled_temporal_dates,
    is_explicitly_after_cutoff,
    is_publication_date_after_cutoff,
    is_text_explicitly_after_cutoff,
)


def test_extract_years():
    assert extract_explicit_years(
        "Report published in 2022"
    ) == [2022]


def test_extract_full_url_date():
    result = extract_explicit_dates(
        "https://example.com/2020/11/02/article"
    )

    assert len(result) == 1

    assert result[
        0
    ].strftime(
        "%Y-%m-%d"
    ) == "2020-11-02"


def test_extract_written_date():
    result = extract_explicit_dates(
        "Published April 28, 2022"
    )

    assert len(result) == 1

    assert result[
        0
    ].strftime(
        "%Y-%m-%d"
    ) == "2022-04-28"


def test_extract_labeled_publication_date():
    result = (
        extract_labeled_temporal_dates(
            "Article\n"
            "Published: April 28, 2022\n"
            "Abstract"
        )
    )

    assert len(result) == 1


def test_future_title_is_rejected():
    assert is_explicitly_after_cutoff(
        title="2022 Report Card",
        url="https://example.com/report.pdf",
        end_date="2020-10-30",
    )


def test_future_year_in_url_is_rejected():
    assert is_explicitly_after_cutoff(
        title="COVID report",
        url="https://example.com/2022/report",
        end_date="2020-10-30",
    )


def test_mixed_year_url_with_future_year_is_rejected():
    assert is_explicitly_after_cutoff(
        title="2022 Report Card",
        url=(
            "https://example.com/"
            "2016/report-2022.pdf"
        ),
        end_date="2020-10-30",
    )


def test_same_year_future_date_is_rejected():
    assert is_explicitly_after_cutoff(
        title="Fact check",
        url=(
            "https://example.com/"
            "2020/11/02/article"
        ),
        end_date="2020-10-30",
    )


def test_same_year_old_date_is_allowed():
    assert not is_explicitly_after_cutoff(
        title="COVID report",
        url=(
            "https://example.com/"
            "2020/09/22/article"
        ),
        end_date="2020-10-30",
    )


def test_old_title_is_allowed():
    assert not is_explicitly_after_cutoff(
        title="COVID report 2020",
        url="https://example.com/report",
        end_date="2020-10-30",
    )


def test_undated_result_is_allowed():
    assert not is_explicitly_after_cutoff(
        title="COVID mortality report",
        url="https://example.com/report",
        end_date="2020-10-30",
    )


def test_post_cutoff_publication_text_is_rejected():
    assert is_text_explicitly_after_cutoff(
        text=(
            "COVID mortality study\n"
            "Published: April 28, 2022\n"
            "This study examines mortality."
        ),
        end_date="2020-10-30",
    )


def test_pre_cutoff_publication_text_is_allowed():
    assert not is_text_explicitly_after_cutoff(
        text=(
            "COVID mortality study\n"
            "Published: June 19, 2020\n"
            "This study examines mortality."
        ),
        end_date="2020-10-30",
    )


def test_unlabeled_future_event_date_does_not_reject():
    assert not is_text_explicitly_after_cutoff(
        text=(
            "The model projects cases "
            "through April 28, 2022."
        ),
        end_date="2020-10-30",
    )


def test_metadata_date_after_cutoff_is_rejected():
    assert is_publication_date_after_cutoff(
        published_date="2021-04-08",
        end_date="2020-10-30",
    )


def test_metadata_date_before_cutoff_is_allowed():
    assert not is_publication_date_after_cutoff(
        published_date="2020-10-22",
        end_date="2020-10-30",
    )
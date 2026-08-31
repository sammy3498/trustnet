from backend.app.evidence.domain import get_publisher_domain


def test_nasa_subdomain():
    assert (
        get_publisher_domain("science.nasa.gov")
        == "nasa.gov"
    )


def test_nasa_url():
    assert (
        get_publisher_domain(
            "https://science.nasa.gov/test"
        )
        == "nasa.gov"
    )


def test_multi_part_suffix():
    assert (
        get_publisher_domain("news.bbc.co.uk")
        == "bbc.co.uk"
    )


def test_regular_domain():
    assert (
        get_publisher_domain("www.reuters.com")
        == "reuters.com"
    )
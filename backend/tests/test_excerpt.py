from backend.app.retrieval.excerpt import (
    build_passage_windows,
    get_relevant_excerpt,
)


def test_passage_respects_character_limit():
    text = (
        "Alpha information about an unrelated "
        "subject. " * 50
    )

    passages = (
        build_passage_windows(
            text=text,
            max_chars=300,
        )
    )

    assert passages

    assert all(
        len(passage) <= 300
        for passage in passages
    )


def test_relevant_excerpt_finds_late_content():
    text = (
        (
            "This section discusses gardening "
            "and flowers without mentioning "
            "astronomy. "
        )
        * 20
        + "\n"
        + (
            "The Earth orbits the Sun once "
            "during each year. "
        )
        * 8
    )

    excerpt = (
        get_relevant_excerpt(
            text=text,
            query=(
                "Earth orbits Sun"
            ),
            max_chars=500,
        )
    )

    lowered = (
        excerpt.lower()
    )

    assert "earth" in lowered
    assert "orbits" in lowered
    assert "sun" in lowered

    assert len(excerpt) <= 500


def test_passage_windows_are_contiguous():
    paragraph_one = (
        "PARAGRAPH_ONE "
        + "alpha " * 25
    )

    paragraph_two = (
        "PARAGRAPH_TWO "
        + "target " * 25
    )

    paragraph_three = (
        "PARAGRAPH_THREE "
        + "target " * 25
    )

    paragraph_four = (
        "PARAGRAPH_FOUR "
        + "omega " * 25
    )

    text = "\n".join(
        [
            paragraph_one,
            paragraph_two,
            paragraph_three,
            paragraph_four,
        ]
    )

    excerpt = (
        get_relevant_excerpt(
            text=text,
            query=(
                "target"
            ),
            max_chars=400,
        )
    )

    assert (
        "PARAGRAPH_TWO"
        in excerpt
        or "PARAGRAPH_THREE"
        in excerpt
    )

    assert not (
        "PARAGRAPH_ONE"
        in excerpt
        and "PARAGRAPH_FOUR"
        in excerpt
    )


def test_empty_text_returns_empty_excerpt():
    assert (
        get_relevant_excerpt(
            text="",
            query="example",
        )
        == ""
    )


def test_short_text_is_preserved():
    text = (
        "Short directly relevant "
        "evidence."
    )

    assert (
        get_relevant_excerpt(
            text=text,
            query="relevant evidence",
        )
        == text
    )
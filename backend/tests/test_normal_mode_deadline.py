import time

from backend.app.llm.pipeline import (
    LLM_ANALYSIS_DEADLINE_RESERVE_SECONDS,
    LLM_ANALYSIS_WALL_TIMEOUT_SECONDS,
    resolve_llm_api_timeout,
    resolve_llm_wall_timeout,
)


def test_llm_wall_timeout_uses_default_without_deadline():
    assert (
        resolve_llm_wall_timeout(None)
        == LLM_ANALYSIS_WALL_TIMEOUT_SECONDS
    )


def test_llm_wall_timeout_respects_remaining_deadline():
    deadline = time.perf_counter() + 5.0

    timeout = resolve_llm_wall_timeout(
        deadline
    )

    assert 4.0 < timeout < 4.5


def test_llm_wall_timeout_never_exceeds_stage_cap():
    deadline = time.perf_counter() + 60.0

    assert (
        resolve_llm_wall_timeout(
            deadline
        )
        == LLM_ANALYSIS_WALL_TIMEOUT_SECONDS
    )


def test_llm_wall_timeout_returns_zero_when_budget_spent():
    deadline = (
        time.perf_counter()
        + LLM_ANALYSIS_DEADLINE_RESERVE_SECONDS
        - 0.1
    )

    assert (
        resolve_llm_wall_timeout(
            deadline
        )
        == 0.0
    )


def test_llm_api_timeout_is_below_wall_timeout():
    wall_timeout = 6.0

    api_timeout = resolve_llm_api_timeout(
        wall_timeout
    )

    assert api_timeout < wall_timeout
    assert api_timeout == 5.75
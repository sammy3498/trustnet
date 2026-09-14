import pytest

import backend.app.llm.pipeline as llm_pipeline
from backend.app.llm.evidence_analysis import (
    EVIDENCE_ANALYSIS_TIMEOUT_SECONDS,
    resolve_evidence_analysis_timeout,
)
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

    assert (
        LLM_ANALYSIS_WALL_TIMEOUT_SECONDS
        == 56.0
    )


def test_llm_wall_timeout_respects_remaining_deadline(
    monkeypatch,
):
    monkeypatch.setattr(
        llm_pipeline.time,
        "perf_counter",
        lambda: 100.0,
    )

    deadline = 105.0

    timeout = resolve_llm_wall_timeout(
        deadline
    )

    assert timeout == pytest.approx(
        3.5
    )


def test_llm_wall_timeout_never_exceeds_stage_cap(
    monkeypatch,
):
    monkeypatch.setattr(
        llm_pipeline.time,
        "perf_counter",
        lambda: 100.0,
    )

    deadline = 200.0

    assert (
        resolve_llm_wall_timeout(
            deadline
        )
        == LLM_ANALYSIS_WALL_TIMEOUT_SECONDS
    )


def test_llm_wall_timeout_returns_zero_when_budget_spent(
    monkeypatch,
):
    monkeypatch.setattr(
        llm_pipeline.time,
        "perf_counter",
        lambda: 100.0,
    )

    deadline = (
        100.0
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
    wall_timeout = 56.0

    api_timeout = resolve_llm_api_timeout(
        wall_timeout
    )

    assert api_timeout == 55.0
    assert api_timeout < wall_timeout


def test_evidence_timeout_defaults_to_extended_window():
    assert (
        resolve_evidence_analysis_timeout(
            None
        )
        == EVIDENCE_ANALYSIS_TIMEOUT_SECONDS
    )

    assert (
        EVIDENCE_ANALYSIS_TIMEOUT_SECONDS
        == 55.0
    )


def test_evidence_timeout_respects_smaller_deadline():
    assert (
        resolve_evidence_analysis_timeout(
            32.5
        )
        == 32.5
    )


def test_evidence_timeout_never_exceeds_cap():
    assert (
        resolve_evidence_analysis_timeout(
            100.0
        )
        == 55.0
    )
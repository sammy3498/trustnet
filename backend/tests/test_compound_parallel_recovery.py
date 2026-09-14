import threading
import time
from types import SimpleNamespace

from backend.app.evidence import (
    compound_fast_path,
)


def resolved_analysis(
    direction: str,
) -> dict:
    return {
        "evidence_assessments": [],
        "subclaim_assessments": [
            {
                "subclaim": "atomic",
                "coverage": "complete",
                "direct_support": (
                    direction == "supports"
                ),
                "direct_contradiction": (
                    direction
                    == "contradicts"
                ),
                "material_context_conflict": False,
                "status": direction,
                "reasoning": "Resolved.",
            }
        ],
        "overall_evidence_direction": (
            direction
        ),
        "material_caveats": [],
        "_fast_path": {
            "effective_evidence_count": 2,
            "effective_high_quality_count": 1,
            "effective_medium_quality_count": 1,
        },
    }


def test_compound_recovery_calls_overlap(
    monkeypatch,
):
    active = 0
    max_active = 0
    lock = threading.Lock()

    def fake_atomic(
        claim,
    ):
        return True

    def fake_probe(
        subclaim,
        evidence,
    ):
        if evidence:
            return (
                resolved_analysis(
                    "supports"
                ),
                1,
            )

        return (
            None,
            1,
        )

    def fake_recovery(
        subclaim,
        evidence,
        end_date=None,
        deadline=None,
    ):
        nonlocal active
        nonlocal max_active

        with lock:
            active += 1
            max_active = max(
                max_active,
                active,
            )

        time.sleep(
            0.05
        )

        with lock:
            active -= 1

        item = SimpleNamespace(
            url=(
                "https://example.com/"
                + subclaim.replace(
                    " ",
                    "-",
                )
            )
        )

        evidence.append(
            item
        )

        return [
            item
        ]

    monkeypatch.setattr(
        compound_fast_path,
        "is_simple_atomic_claim",
        fake_atomic,
    )

    monkeypatch.setattr(
        compound_fast_path,
        "run_fast_probe",
        fake_probe,
    )

    monkeypatch.setattr(
        compound_fast_path,
        "recover_atomic_subclaim_evidence",
        fake_recovery,
    )

    result = (
        compound_fast_path
        .try_compound_fast_evidence_analysis(
            claim="A and B",
            subclaims=[
                "A",
                "B",
            ],
            evidence=[],
        )
    )

    assert max_active == 2

    assert (
        result[
            "analysis"
        ][
            "overall_evidence_direction"
        ]
        == "supports"
    )

def test_resolve_atomic_subclaim_stays_sequential(
    monkeypatch,
):
    calls = []

    def fake_probe(
        subclaim,
        evidence,
    ):
        calls.append(
            (
                "probe",
                len(
                    evidence
                ),
            )
        )

        if evidence:
            return (
                resolved_analysis(
                    "supports"
                ),
                2,
            )

        return (
            None,
            2,
        )

    def fake_recovery(
        subclaim,
        evidence,
        end_date=None,
        deadline=None,
    ):
        calls.append(
            (
                "recover",
                len(
                    evidence
                ),
            )
        )

        item = SimpleNamespace(
            url="https://example.com/a"
        )

        evidence.append(
            item
        )

        return [
            item
        ]

    monkeypatch.setattr(
        compound_fast_path,
        "run_fast_probe",
        fake_probe,
    )

    monkeypatch.setattr(
        compound_fast_path,
        "recover_atomic_subclaim_evidence",
        fake_recovery,
    )

    result = (
        compound_fast_path
        .resolve_atomic_subclaim(
            subclaim="A",
            evidence=[],
        )
    )

    assert (
        result[
            "analysis"
        ][
            "overall_evidence_direction"
        ]
        == "supports"
    )

    assert calls == [
        (
            "probe",
            0,
        ),
        (
            "recover",
            0,
        ),
        (
            "probe",
            1,
        ),
    ]

import json
from pathlib import Path

from pydantic import (
    ValidationError,
)

from backend.app.evidence.direction import (
    align_evidence_strength_score,
    resolve_overall_evidence_direction,
)
from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.client import (
    get_llm_response,
)
from backend.app.models.evidence_analysis import (
    EvidenceAnalysisResponseV5,
)
from backend.app.retrieval.excerpt import (
    get_relevant_excerpt,
)


PROMPT_PATH = Path(
    "prompts/evidence_analysis_v5.txt"
)

EVIDENCE_ANALYSIS_SCHEMA = (
    EvidenceAnalysisResponseV5
    .model_json_schema()
)

EVIDENCE_ANALYSIS_TIMEOUT_SECONDS = (
    10.5
)

NORMAL_MODE_EVIDENCE_MAX_CHARS = (
    1200
)


def resolve_evidence_analysis_timeout(
    timeout_seconds: float | None,
) -> float:
    if timeout_seconds is None:
        return EVIDENCE_ANALYSIS_TIMEOUT_SECONDS

    return min(
        EVIDENCE_ANALYSIS_TIMEOUT_SECONDS,
        max(
            0.1,
            float(
                timeout_seconds
            ),
        ),
    )


def load_evidence_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def apply_subclaim_consistency(
    analysis: dict,
) -> dict:
    subclaims = analysis[
        "subclaim_assessments"
    ]

    for subclaim in subclaims:
        original_status = (
            subclaim[
                "status"
            ]
        )

        subclaim[
            "llm_status"
        ] = original_status

        coverage = (
            subclaim[
                "coverage"
            ]
        )

        direct_support = bool(
            subclaim[
                "direct_support"
            ]
        )

        direct_contradiction = bool(
            subclaim[
                "direct_contradiction"
            ]
        )

        context_conflict = bool(
            subclaim[
                "material_context_conflict"
            ]
        )

        if (
            direct_support
            and direct_contradiction
        ):
            resolved_status = (
                "mixed"
            )

        elif (
            direct_support
            and context_conflict
        ):
            resolved_status = (
                "mixed"
            )

        elif direct_contradiction:
            resolved_status = (
                "contradicts"
            )

        elif (
            direct_support
            and coverage
            == "complete"
        ):
            resolved_status = (
                "supports"
            )

        else:
            resolved_status = (
                "insufficient"
            )

        subclaim[
            "status"
        ] = resolved_status

        subclaim[
            "status_was_resolved"
        ] = (
            resolved_status
            != original_status
        )

    return analysis


def apply_v4_subclaim_consistency(
    analysis: dict,
) -> dict:
    return (
        apply_subclaim_consistency(
            analysis
        )
    )


def apply_v5_subclaim_consistency(
    analysis: dict,
) -> dict:
    return (
        apply_subclaim_consistency(
            analysis
        )
    )


def apply_direction_consistency(
    analysis: dict,
) -> dict:
    original_direction = (
        analysis[
            "overall_evidence_direction"
        ]
    )

    resolved_direction = (
        resolve_overall_evidence_direction(
            subclaim_assessments=(
                analysis[
                    "subclaim_assessments"
                ]
            ),
            fallback_direction=(
                original_direction
            ),
        )
    )

    analysis[
        "llm_overall_evidence_direction"
    ] = original_direction

    analysis[
        "overall_evidence_direction"
    ] = resolved_direction

    analysis[
        "direction_was_resolved"
    ] = (
        resolved_direction
        != original_direction
    )

    if (
        "evidence_strength_score"
        in analysis
    ):
        original_score = (
            analysis[
                "evidence_strength_score"
            ]
        )

        resolved_score = (
            align_evidence_strength_score(
                score=original_score,
                original_direction=(
                    original_direction
                ),
                resolved_direction=(
                    resolved_direction
                ),
            )
        )

        analysis[
            "llm_evidence_strength_score"
        ] = original_score

        analysis[
            "evidence_strength_score"
        ] = resolved_score

    return analysis


def analyze_evidence_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
    timeout_seconds: float | None = None,
) -> dict:
    system_prompt = (
        load_evidence_prompt()
    )

    evidence_parts = []

    for item in evidence:
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=claim,
                max_chars=(
                    NORMAL_MODE_EVIDENCE_MAX_CHARS
                ),
            )
        )

        evidence_parts.append(
            f"""
Title: {item.title}
URL: {item.url}
Quality: {item.quality}
Origin: {item.evidence_origin}
Evidence:
{excerpt}
"""
        )

    evidence_text = (
        "\n\n".join(
            evidence_parts
        )
    )

    user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}
"""

    raw_response = (
        get_llm_response(
            system_prompt=(
                system_prompt
            ),
            user_prompt=(
                user_prompt
            ),
            response_schema=(
                EVIDENCE_ANALYSIS_SCHEMA
            ),
            response_schema_name=(
                "trustnet_evidence_analysis_v5"
            ),
            timeout_seconds=(
                resolve_evidence_analysis_timeout(
                    timeout_seconds
                )
            ),
        )
    )

    try:
        data = json.loads(
            raw_response
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid "
            f"JSON:\n{raw_response}"
        ) from exc

    try:
        validated = (
            EvidenceAnalysisResponseV5
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "LLM returned invalid "
            "evidence analysis:\n"
            f"{exc}"
        ) from exc

    analysis = (
        validated.model_dump()
    )

    analysis = (
        apply_v5_subclaim_consistency(
            analysis
        )
    )

    return (
        apply_direction_consistency(
            analysis
        )
    )
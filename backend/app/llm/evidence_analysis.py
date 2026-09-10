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
    EvidenceAnalysisResponse,
)
from backend.app.retrieval.excerpt import (
    get_relevant_excerpt,
)


PROMPT_PATH = Path(
    "prompts/evidence_analysis_v2.txt"
)

EVIDENCE_ANALYSIS_SCHEMA = (
    EvidenceAnalysisResponse
    .model_json_schema()
)


def load_evidence_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def apply_direction_consistency(
    analysis: dict,
) -> dict:
    original_direction = (
        analysis[
            "overall_evidence_direction"
        ]
    )

    original_score = (
        analysis[
            "evidence_strength_score"
        ]
    )

    subclaims = (
        analysis[
            "subclaim_assessments"
        ]
    )

    resolved_direction = (
        resolve_overall_evidence_direction(
            subclaim_assessments=(
                subclaims
            ),
            fallback_direction=(
                original_direction
            ),
        )
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
        "llm_overall_evidence_direction"
    ] = original_direction

    analysis[
        "llm_evidence_strength_score"
    ] = original_score

    analysis[
        "overall_evidence_direction"
    ] = resolved_direction

    analysis[
        "evidence_strength_score"
    ] = resolved_score

    analysis[
        "direction_was_resolved"
    ] = (
        resolved_direction
        != original_direction
    )

    return analysis


def analyze_evidence_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
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
                max_chars=3000,
            )
        )

        evidence_parts.append(
            f"""
        Title: {item.title}
        URL: {item.url}
        Source: {item.source}
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
                "trustnet_evidence_analysis"
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
            EvidenceAnalysisResponse
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

    return (
        apply_direction_consistency(
            analysis
        )
    )
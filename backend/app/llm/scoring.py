import json
from pathlib import Path

from pydantic import (
    ValidationError,
)

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.client import (
    get_llm_response,
)
from backend.app.models.credibility_assessment import (
    CredibilityAssessmentResponse,
)
from backend.app.retrieval.excerpt import (
    DEFAULT_PASSAGE_MAX_CHARS,
    get_relevant_excerpt,
)


PROMPT_PATH = Path(
    "prompts/credibility_scoring_v2.txt"
)

CREDIBILITY_ASSESSMENT_SCHEMA = (
    CredibilityAssessmentResponse
    .model_json_schema()
)

CREDIBILITY_SCORING_TIMEOUT_SECONDS = (
    5.0
)


def load_scoring_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def score_claim_with_llm(
    claim: str,
    evidence: list[
        RetrievedEvidence
    ],
) -> dict:
    system_prompt = (
        load_scoring_prompt()
    )

    evidence_parts = []

    for item in evidence:
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=claim,
                max_chars=(
                    DEFAULT_PASSAGE_MAX_CHARS
                ),
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
                CREDIBILITY_ASSESSMENT_SCHEMA
            ),
            response_schema_name=(
                "trustnet_credibility_assessment_v2"
            ),
            timeout_seconds=(
                CREDIBILITY_SCORING_TIMEOUT_SECONDS
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
            CredibilityAssessmentResponse
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "LLM returned invalid "
            "credibility assessment:\n"
            f"{exc}"
        ) from exc

    return validated.model_dump()
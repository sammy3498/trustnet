import json
from pathlib import Path

from pydantic import ValidationError

from backend.app.evidence.models import (
    RetrievedEvidence,
)
from backend.app.llm.client import (
    get_llm_response,
)
from backend.app.models.schemas import (
    LLMScoringResponse,
)
from backend.app.retrieval.excerpt import (
    get_relevant_excerpt,
)


PROMPT_PATH = Path(
    "prompts/credibility_scoring_v1.txt"
)

LLM_SCORING_SCHEMA = (
    LLMScoringResponse
    .model_json_schema()
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
    timeout_seconds: float | None = None,
) -> dict:
    system_prompt = (
        load_scoring_prompt()
    )

    evidence_parts = []

    for item in evidence:
        excerpt = (
            get_relevant_excerpt(
                text=item.text,
                query=(item.matched_query or claim),
                max_chars=1500,
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
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=(
                LLM_SCORING_SCHEMA
            ),
            response_schema_name=(
                "trustnet_credibility_scoring"
            ),
            timeout_seconds=(
                timeout_seconds
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
            LLMScoringResponse
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "LLM returned JSON with "
            "an invalid structure:\n"
            f"{exc}"
        ) from exc

    return validated.model_dump()
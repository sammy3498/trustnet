import json
from pathlib import Path

from pydantic import ValidationError

from backend.app.evidence.models import RetrievedEvidence
from backend.app.llm.client import get_llm_response
from backend.app.models.evidence_analysis import EvidenceAnalysisResponse
from backend.app.retrieval.excerpt import get_relevant_excerpt

PROMPT_PATH = Path("prompts/evidence_analysis_v1.txt")


def load_evidence_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def analyze_evidence_with_llm(
    claim: str,
    evidence: list[RetrievedEvidence],
) -> dict:
    system_prompt = load_evidence_prompt()

    evidence_parts = []

    for item in evidence:
        excerpt = get_relevant_excerpt(
            text=item.text,
            query=claim,
            max_chars=3000,
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

    evidence_text = "\n\n".join(evidence_parts)

    user_prompt = f"""
Claim:
{claim}

Evidence:
{evidence_text}
"""

    raw_response = get_llm_response(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON:\n{raw_response}"
        ) from exc

    try:
        validated = EvidenceAnalysisResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"LLM returned invalid evidence analysis:\n{exc}"
        ) from exc

    return validated.model_dump()
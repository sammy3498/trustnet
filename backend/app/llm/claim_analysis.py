import json
from pathlib import Path

from backend.app.llm.client import get_llm_response
from backend.app.models.claim_analysis import ClaimAnalysisResponse


PROMPT_PATH = Path("prompts/claim_extraction_v1.txt")


def analyze_claim_clarity(claim: str) -> dict:
    system_prompt = PROMPT_PATH.read_text(encoding="utf-8")

    user_prompt = f"""
Analyze this claim:

{claim}

Return JSON exactly in this format:

{{
  "normalized_claim": "clear standalone factual claim",
  "is_clear": true,
  "reason": "brief explanation"
}}
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

    validated = ClaimAnalysisResponse.model_validate(data)

    return validated.model_dump()
import json
from pathlib import Path

from pydantic import ValidationError

from backend.app.llm.client import get_llm_response
from backend.app.models.claim_analysis import ClaimPreparationResponse


PROMPT_PATH = Path("prompts/claim_preparation_v1.txt")


def load_claim_preparation_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prepare_claim(
    claim: str,
    context: str | None = None,
):
    system_prompt = load_claim_preparation_prompt()

    if context:
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Claim:\n{claim}"
        )
    else:
        user_prompt = claim

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
        validated = ClaimPreparationResponse.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            f"LLM returned invalid claim preparation:\n{exc}"
        ) from exc

    return validated.model_dump()
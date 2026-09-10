import json
from pathlib import Path

from pydantic import (
    ValidationError,
)

from backend.app.llm.client import (
    get_llm_response,
)
from backend.app.models.claim_analysis import (
    ClaimPreparationResponse,
)


PROMPT_PATH = Path(
    "prompts/claim_preparation_v2.txt"
)

CLAIM_PREPARATION_SCHEMA = (
    ClaimPreparationResponse
    .model_json_schema()
)


def load_claim_preparation_prompt() -> str:
    return PROMPT_PATH.read_text(
        encoding="utf-8"
    )


def prepare_claim(
    claim: str,
    context: str | None = None,
):
    system_prompt = (
        load_claim_preparation_prompt()
    )

    if context:
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Claim:\n{claim}"
        )
    else:
        user_prompt = (
            f"Claim:\n{claim}"
        )

    raw_response = (
        get_llm_response(
            system_prompt=(
                system_prompt
            ),
            user_prompt=(
                user_prompt
            ),
            response_schema=(
                CLAIM_PREPARATION_SCHEMA
            ),
            response_schema_name=(
                "trustnet_claim_preparation"
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
            ClaimPreparationResponse
            .model_validate(
                data
            )
        )

    except ValidationError as exc:
        raise ValueError(
            "LLM returned invalid "
            "claim preparation:\n"
            f"{exc}"
        ) from exc

    return validated.model_dump()
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DEFAULT_LLM_TIMEOUT_SECONDS = 11.0
MAX_LLM_TIMEOUT_SECONDS = 11.0

DEFAULT_REASONING_EFFORT = "none"
DEFAULT_MAX_OUTPUT_TOKENS = 900


api_key = os.getenv(
    "LLM_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "LLM_API_KEY is not configured."
    )


client = OpenAI(
    api_key=api_key,
    max_retries=0,
    timeout=(
        DEFAULT_LLM_TIMEOUT_SECONDS
    ),
)


def resolve_timeout_seconds(
    timeout_seconds: float | None,
) -> float:
    if timeout_seconds is None:
        return (
            DEFAULT_LLM_TIMEOUT_SECONDS
        )

    requested = max(
        0.1,
        float(
            timeout_seconds
        ),
    )

    return min(
        requested,
        MAX_LLM_TIMEOUT_SECONDS,
    )


def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None = None,
    response_schema_name: str = (
        "structured_response"
    ),
    timeout_seconds: float | None = None,
) -> str:
    model = os.getenv(
        "LLM_MODEL"
    )

    if not model:
        raise RuntimeError(
            "LLM_MODEL is not configured."
        )

    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
        "reasoning": {
            "effort": (
                DEFAULT_REASONING_EFFORT
            )
        },
        "max_output_tokens": (
            DEFAULT_MAX_OUTPUT_TOKENS
        ),
    }

    if response_schema is not None:
        request["text"] = {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": (
                    response_schema_name
                ),
                "schema": (
                    response_schema
                ),
                "strict": True,
            },
        }

    else:
        request["text"] = {
            "verbosity": "low"
        }

    effective_timeout = (
        resolve_timeout_seconds(
            timeout_seconds
        )
    )

    response = (
        client
        .with_options(
            timeout=(
                effective_timeout
            )
        )
        .responses
        .create(
            **request
        )
    )

    return response.output_text
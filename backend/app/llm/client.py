import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv(
    "LLM_API_KEY"
)

if not api_key:
    raise RuntimeError(
        "LLM_API_KEY is not configured."
    )

client = OpenAI(
    api_key=api_key
)


def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None = None,
    response_schema_name: str = (
        "structured_response"
    ),
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
    }

    if response_schema is not None:
        request["text"] = {
            "format": {
                "type": "json_schema",
                "name": (
                    response_schema_name
                ),
                "schema": (
                    response_schema
                ),
                "strict": True,
            }
        }

    response = (
        client.responses.create(
            **request
        )
    )

    return response.output_text
import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

api_key = os.getenv("LLM_API_KEY")

if not api_key:
    raise RuntimeError("LLM_API_KEY is not configured.")

client = OpenAI(api_key=api_key)


def get_llm_response(system_prompt: str, user_prompt: str) -> str:
    model = os.getenv("LLM_MODEL")

    if not model:
        raise RuntimeError("LLM_MODEL is not configured.")

    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=user_prompt,
    )

    return response.output_text
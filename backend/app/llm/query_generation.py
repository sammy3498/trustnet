import json

from backend.app.llm.client import get_llm_response
from backend.app.models.query_generation import QueryGenerationResponse


SYSTEM_PROMPT = """
You generate concise web search queries for factual claim verification.

Given a factual claim, produce one search query designed to find strong,
relevant evidence that can confirm or contradict it.

Rules:
- Preserve the core meaning of the claim.
- Include important entities, dates, quantities, and technical terms.
- Do not answer the claim.
- Do not assume the claim is true.
- Return JSON only.

Format:
{
  "query": "search query"
}
"""


def generate_search_query(claim: str) -> str:
    raw_response = get_llm_response(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=claim,
    )

    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON:\n{raw_response}"
        ) from exc

    validated = QueryGenerationResponse.model_validate(data)

    return validated.query
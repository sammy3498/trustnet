import json
import os
import re
import time
from functools import lru_cache
from concurrent.futures import (
    ThreadPoolExecutor,
    TimeoutError as FuturesTimeoutError,
)

from dotenv import load_dotenv
from openai import OpenAI
from openai import RateLimitError


load_dotenv()


class TransientLLMError(
    TimeoutError
):
    pass


def get_provider() -> str:
    provider = os.getenv(
        "LLM_PROVIDER",
        "openai",
    ).strip().lower()

    if provider not in {
        "openai",
        "openrouter",
        "gemini",
    }:
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER: {provider}"
        )

    return provider



@lru_cache(maxsize=4)
def _cached_client(provider: str, api_key: str) -> OpenAI:
    """Reuse the HTTP connection pool across LLM calls and requests."""
    base_urls = {
        "openrouter": "https://openrouter.ai/api/v1",
        "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    }
    settings = {"api_key": api_key, "max_retries": 0}
    if provider in base_urls:
        settings["base_url"] = base_urls[provider]
    return OpenAI(**settings)


def get_client_and_model():
    provider = get_provider()

    if provider == "openai":
        api_key = os.getenv(
            "OPENAI_API_KEY"
        )

        model = os.getenv(
            "OPENAI_MODEL"
        )

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured."
            )

        if not model:
            raise RuntimeError(
                "OPENAI_MODEL is not configured."
            )

        client = _cached_client(provider, api_key)

        return (
            provider,
            client,
            model,
        )

    if provider == "openrouter":
        api_key = os.getenv(
            "OPENROUTER_API_KEY"
        )

        model = os.getenv(
            "OPENROUTER_MODEL"
        )

        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured."
            )

        if not model:
            raise RuntimeError(
                "OPENROUTER_MODEL is not configured."
            )

        client = _cached_client(provider, api_key)

        return (
            provider,
            client,
            model,
        )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    model = os.getenv(
        "GEMINI_MODEL"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    if not model:
        raise RuntimeError(
            "GEMINI_MODEL is not configured."
        )

    client = _cached_client(provider, api_key)

    return (
        provider,
        client,
        model,
    )


def strip_markdown_fences(
    text: str,
) -> str:
    value = text.strip()

    if value.startswith("```"):
        value = re.sub(
            r"^```(?:json)?\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )

        value = re.sub(
            r"\s*```$",
            "",
            value,
        )

    return value.strip()


def validate_json_text(
    text: str,
) -> str:
    cleaned = strip_markdown_fences(
        text
    )

    json.loads(
        cleaned
    )

    return cleaned


def build_chat_messages(
    system_prompt: str,
    user_prompt: str,
    force_json: bool = False,
):
    system_text = system_prompt

    if force_json:
        system_text = (
            system_prompt
            + "\n\n"
            + "Return only valid raw JSON. "
            + "Do not use Markdown code fences. "
            + "Do not include explanatory text "
            + "before or after the JSON."
        )

    return [
        {
            "role": "system",
            "content": system_text,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def run_with_wall_clock_timeout(
    function,
    timeout_seconds: float | None,
    stage_name: str,
):
    if timeout_seconds is None:
        return function()

    timeout_seconds = max(
        0.25,
        timeout_seconds,
    )

    executor = ThreadPoolExecutor(
        max_workers=1
    )

    future = executor.submit(
        function
    )

    try:
        return future.result(
            timeout=timeout_seconds
        )

    except FuturesTimeoutError as exc:
        future.cancel()

        raise TimeoutError(
            f"{stage_name} exceeded its "
            "wall-clock time budget."
        ) from exc

    finally:
        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


def get_remaining_timeout(
    started_at: float,
    total_timeout: float | None,
) -> float | None:
    if total_timeout is None:
        return None

    elapsed = (
        time.perf_counter()
        - started_at
    )

    remaining = (
        total_timeout
        - elapsed
    )

    if remaining <= 0.25:
        return None

    return remaining


def _is_transient_provider_error(
    exc: BaseException,
) -> bool:
    status_code = getattr(
        exc,
        "status_code",
        None,
    )

    if status_code in {
        500,
        502,
        503,
        504,
    }:
        return True

    return exc.__class__.__name__ in {
        "InternalServerError",
        "APIConnectionError",
        "ServiceUnavailableError",
    }


def _rate_limit_message(
    provider: str,
) -> str:
    return (
        f"{provider.title()} model is "
        "currently rate-limited. Try again "
        "shortly or switch models/providers."
    )


def _request_with_transient_retry(
    request_function,
    provider: str,
    started_at: float,
    total_timeout: float | None,
    stage_name: str,
    max_attempts: int = 2,
):
    last_error = None

    for attempt in range(
        max_attempts
    ):
        remaining = (
            get_remaining_timeout(
                started_at=started_at,
                total_timeout=(
                    total_timeout
                ),
            )
        )

        if (
            total_timeout is not None
            and remaining is None
        ):
            raise TimeoutError(
                f"{stage_name} time budget "
                "was exhausted before the "
                "request could start."
            )

        try:
            return request_function(
                remaining
            )

        except RateLimitError as exc:
            raise RuntimeError(
                _rate_limit_message(
                    provider
                )
            ) from exc

        except Exception as exc:
            if not (
                _is_transient_provider_error(
                    exc
                )
            ):
                raise

            last_error = exc

            if attempt + 1 >= max_attempts:
                break

            remaining_after_error = (
                get_remaining_timeout(
                    started_at=started_at,
                    total_timeout=(
                        total_timeout
                    ),
                )
            )

            if (
                total_timeout is not None
                and (
                    remaining_after_error
                    is None
                    or remaining_after_error
                    < 0.75
                )
            ):
                break

    raise TransientLLMError(
        f"{provider.title()} temporarily "
        "failed after one retry."
    ) from last_error


def openai_responses_request(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None,
    response_schema_name: str,
    timeout_seconds: float | None,
    use_schema: bool,
):
    request = {
        "model": model,
        "instructions": system_prompt,
        "input": user_prompt,
    }

    if (
        use_schema
        and response_schema is not None
    ):
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

    if timeout_seconds is not None:
        request["timeout"] = max(
            0.25,
            timeout_seconds,
        )

    def execute():
        return (
            client.responses.create(
                **request
            )
        )

    response = (
        run_with_wall_clock_timeout(
            function=execute,
            timeout_seconds=(
                timeout_seconds
            ),
            stage_name=(
                "OpenAI Responses request"
            ),
        )
    )

    text = getattr(
        response,
        "output_text",
        None,
    )

    if not text:
        return None

    return text


def compatible_chat_request(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None,
    response_schema_name: str,
    timeout_seconds: float | None,
    use_schema: bool,
):
    messages = (
        build_chat_messages(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            force_json=(
                response_schema is not None
            ),
        )
    )

    request = {
        "model": model,
        "messages": messages,
    }

    if (
        use_schema
        and response_schema is not None
    ):
        request[
            "response_format"
        ] = {
            "type": "json_schema",
            "json_schema": {
                "name": (
                    response_schema_name
                ),
                "strict": True,
                "schema": (
                    response_schema
                ),
            },
        }

    if timeout_seconds is not None:
        request["timeout"] = max(
            0.25,
            timeout_seconds,
        )

    def execute():
        return (
            client.chat.completions.create(
                **request
            )
        )

    response = (
        run_with_wall_clock_timeout(
            function=execute,
            timeout_seconds=(
                timeout_seconds
            ),
            stage_name=(
                "Compatible chat request"
            ),
        )
    )

    content = (
        response.choices[
            0
        ].message.content
    )

    if not content:
        return None

    return content


def _provider_request_function(
    provider: str,
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None,
    response_schema_name: str,
    use_schema: bool,
):
    def request(
        timeout_seconds: float | None,
    ):
        if provider == "openai":
            return (
                openai_responses_request(
                    client=client,
                    model=model,
                    system_prompt=(
                        system_prompt
                    ),
                    user_prompt=(
                        user_prompt
                    ),
                    response_schema=(
                        response_schema
                    ),
                    response_schema_name=(
                        response_schema_name
                    ),
                    timeout_seconds=(
                        timeout_seconds
                    ),
                    use_schema=(
                        use_schema
                    ),
                )
            )

        return (
            compatible_chat_request(
                client=client,
                model=model,
                system_prompt=(
                    system_prompt
                ),
                user_prompt=(
                    user_prompt
                ),
                response_schema=(
                    response_schema
                ),
                response_schema_name=(
                    response_schema_name
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
                use_schema=(
                    use_schema
                ),
            )
        )

    return request


def get_llm_response(
    system_prompt: str,
    user_prompt: str,
    response_schema: dict | None = None,
    response_schema_name: str = (
        "structured_response"
    ),
    timeout_seconds: float | None = None,
) -> str:
    started_at = time.perf_counter()

    (
        provider,
        client,
        model,
    ) = get_client_and_model()

    primary_request = (
        _provider_request_function(
            provider=provider,
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=(
                response_schema
            ),
            response_schema_name=(
                response_schema_name
            ),
            use_schema=True,
        )
    )

    primary_response = (
        _request_with_transient_retry(
            request_function=(
                primary_request
            ),
            provider=provider,
            started_at=started_at,
            total_timeout=(
                timeout_seconds
            ),
            stage_name=(
                "Structured LLM request"
            ),
            max_attempts=2,
        )
    )

    if primary_response:
        if response_schema is None:
            return primary_response

        try:
            return validate_json_text(
                primary_response
            )

        except json.JSONDecodeError:
            pass

    fallback_timeout = (
        get_remaining_timeout(
            started_at=started_at,
            total_timeout=(
                timeout_seconds
            ),
        )
    )

    if (
        timeout_seconds is not None
        and fallback_timeout is None
    ):
        raise TimeoutError(
            "LLM time budget was exhausted "
            "before the JSON fallback request "
            "could start."
        )

    fallback_request = (
        _provider_request_function(
            provider=provider,
            client=client,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=None,
            response_schema_name=(
                response_schema_name
            ),
            use_schema=False,
        )
    )

    try:
        fallback_response = (
            fallback_request(
                fallback_timeout
            )
        )

    except RateLimitError as exc:
        raise RuntimeError(
            _rate_limit_message(
                provider
            )
        ) from exc

    except Exception as exc:
        if _is_transient_provider_error(
            exc
        ):
            raise TransientLLMError(
                f"{provider.title()} temporarily "
                "failed during the JSON "
                "fallback request."
            ) from exc

        raise

    if not fallback_response:
        raise RuntimeError(
            f"{provider.title()} returned an "
            "empty response."
        )

    if response_schema is None:
        return fallback_response

    try:
        return validate_json_text(
            fallback_response
        )

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{provider.title()} returned "
            "invalid JSON after structured-"
            "output fallback."
        ) from exc

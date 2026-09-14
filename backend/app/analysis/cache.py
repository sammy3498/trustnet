import copy
import time
from collections import OrderedDict
from threading import RLock


CACHE_TTL_SECONDS = 300.0
MAX_CACHE_ENTRIES = 128


_cache: OrderedDict[
    tuple,
    tuple[
        float,
        dict,
    ],
] = OrderedDict()

_cache_lock = RLock()


def normalize_cache_text(
    text: str | None,
) -> str:
    if not text:
        return ""

    return " ".join(
        text.split()
    ).strip().lower()


def build_analysis_cache_key(
    claim: str,
    search_results: int,
    top_k: int,
    context: str | None = None,
    retrieval_end_date: str | None = None,
) -> tuple:
    return (
        normalize_cache_text(
            claim
        ),
        normalize_cache_text(
            context
        ),
        int(
            search_results
        ),
        int(
            top_k
        ),
        (
            retrieval_end_date
            or ""
        ),
    )


def remove_expired_entries(
    now: float,
) -> None:
    expired_keys = [
        key
        for key, (
            expires_at,
            _,
        ) in _cache.items()
        if expires_at <= now
    ]

    for key in expired_keys:
        _cache.pop(
            key,
            None,
        )


def get_cached_analysis(
    key: tuple,
) -> dict | None:
    now = time.monotonic()

    with _cache_lock:
        remove_expired_entries(
            now
        )

        cached = _cache.get(
            key
        )

        if cached is None:
            return None

        expires_at, result = cached

        if expires_at <= now:
            _cache.pop(
                key,
                None,
            )

            return None

        _cache.move_to_end(
            key
        )

        return copy.deepcopy(
            result
        )


def set_cached_analysis(
    key: tuple,
    result: dict,
    ttl_seconds: float = (
        CACHE_TTL_SECONDS
    ),
) -> None:
    if ttl_seconds <= 0:
        return

    now = time.monotonic()

    expires_at = (
        now
        + ttl_seconds
    )

    with _cache_lock:
        remove_expired_entries(
            now
        )

        _cache[
            key
        ] = (
            expires_at,
            copy.deepcopy(
                result
            ),
        )

        _cache.move_to_end(
            key
        )

        while (
            len(
                _cache
            )
            > MAX_CACHE_ENTRIES
        ):
            _cache.popitem(
                last=False
            )


def clear_analysis_cache() -> None:
    with _cache_lock:
        _cache.clear()
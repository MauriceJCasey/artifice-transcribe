# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Async retry decorator with exponential backoff for transient API failures.

Transcribe's model calls are ``async def`` (the OpenAI SDK's async client), so
this is the async counterpart of OCR's synchronous ``_retry.py``: it ``await``s
the wrapped call and sleeps with ``asyncio.sleep`` rather than blocking the
event loop on every retry.
"""

import asyncio
import functools
from collections.abc import Awaitable, Callable
from typing import TypeVar

from openai import APIConnectionError, APITimeoutError

from artifice_transcribe._logging import get_logger

log = get_logger("retry")

T = TypeVar("T")

# The OpenAI SDK raises its own exception types for connection and timeout
# failures rather than the builtins OCR's sync decorator catches. Retrying on
# the SDK's actual types is more correct than guessing at builtins it may not
# raise. ``APITimeoutError`` subclasses ``APIConnectionError``; both are listed
# explicitly to make the intent — connection drop *and* timeout — readable.
_DEFAULT_RETRYABLE: tuple[type[Exception], ...] = (APIConnectionError, APITimeoutError)


def retry(
    max_attempts: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable_exceptions: tuple[type[Exception], ...] = _DEFAULT_RETRYABLE,
    label: str = "",
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that retries an async function on transient failures.

    Uses exponential backoff: delay = min(base_delay * 2^attempt, max_delay).

    Args:
        max_attempts: Total number of attempts (1 = no retry, 4 = up to 3 retries).
        base_delay: Initial delay in seconds before first retry.
        max_delay: Cap on backoff delay.
        retryable_exceptions: Exception types that trigger a retry.
        label: Label for log messages (defaults to function name).
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        fn_label = label or fn.__name__

        @functools.wraps(fn)
        async def wrapper(*args, **kwargs) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except retryable_exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        log.error(
                            "%s failed after %d attempts: %s",
                            fn_label,
                            max_attempts,
                            exc,
                        )
                        raise
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    log.warning(
                        "%s attempt %d/%d failed (%s), retrying in %.1fs...",
                        fn_label,
                        attempt,
                        max_attempts,
                        exc.__class__.__name__,
                        delay,
                    )
                    await asyncio.sleep(delay)
            assert last_exc is not None  # unreachable; keeps type checkers happy
            raise last_exc

        return wrapper

    return decorator

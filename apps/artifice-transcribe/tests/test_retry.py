# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The async retry decorator must retry transient failures without blocking.

Transcribe's model calls are ``async def``, so a ``time.sleep``-based decorator
would block the event loop on every retry. These tests exercise the async
version with a zero backoff so nothing actually sleeps for seconds.
"""

import pytest
from artifice_transcribe import _logging
from artifice_transcribe._retry import retry
from openai import APIConnectionError, APITimeoutError


@pytest.fixture(autouse=True)
def _isolated_retry_logging(tmp_path, monkeypatch):
    """Keep the decorator's retry warnings out of a real developer's log."""
    monkeypatch.setenv(_logging.LOG_DIR_ENV, str(tmp_path / "logs"))
    _logging.reset()
    yield
    _logging.reset()


async def test_retry_succeeds_on_first_attempt():
    call_count = 0

    @retry(max_attempts=3, base_delay=0, label="test")
    async def succeed():
        nonlocal call_count
        call_count += 1
        return "ok"

    result = await succeed()
    assert result == "ok"
    assert call_count == 1


async def test_retry_retries_on_failure_then_succeeds():
    call_count = 0

    @retry(max_attempts=3, base_delay=0, label="test")
    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise APIConnectionError(message="transient", request=None)
        return "recovered"

    result = await flaky()
    assert result == "recovered"
    assert call_count == 3


async def test_retry_retries_on_timeout():
    """``APITimeoutError`` subclasses ``APIConnectionError`` but is retried too."""
    call_count = 0

    @retry(max_attempts=2, base_delay=0, label="test")
    async def times_out_once():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise APITimeoutError(request=None)
        return "ok"

    result = await times_out_once()
    assert result == "ok"
    assert call_count == 2


async def test_retry_raises_after_max_attempts():
    call_count = 0

    @retry(max_attempts=2, base_delay=0, label="test")
    async def always_fail():
        nonlocal call_count
        call_count += 1
        raise APIConnectionError(message="permanent", request=None)

    with pytest.raises(APIConnectionError):
        await always_fail()
    assert call_count == 2


async def test_retry_ignores_non_retryable_exceptions():
    call_count = 0

    @retry(max_attempts=3, base_delay=0, label="test")
    async def value_error():
        nonlocal call_count
        call_count += 1
        raise ValueError("not retryable")

    with pytest.raises(ValueError):
        await value_error()
    assert call_count == 1

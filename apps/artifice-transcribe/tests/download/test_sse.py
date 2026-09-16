# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SSE queue behaviour and wire-format tests."""

import contextlib
import threading
from pathlib import Path
from unittest.mock import patch

from artifice_transcribe.main import app
from artifice_transcribe.services.download import record_consent
from httpx import ASGITransport, AsyncClient

# -- SSE queue behaviour -------------------------------------------------------


def test_per_client_queues_are_isolated(clean_manager, clean_consent):
    """Multiple subscribe_events calls produce distinct queues."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    with patch.object(
        dlmod, "_download_with_progress", return_value=(Path("/fake/path"), threading.Thread())
    ):
        clean_manager.start_download("whisper-large-v3")

    q1 = clean_manager.subscribe_events("whisper-large-v3")
    q2 = clean_manager.subscribe_events("whisper-large-v3")

    assert q1 is not q2, "Each subscriber should get its own queue"


def test_unsubscribe_removes_queue(clean_manager, clean_consent):
    """unsubscribe_events removes a queue so it no longer receives events."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    with patch.object(
        dlmod, "_download_with_progress", return_value=(Path("/fake/path"), threading.Thread())
    ):
        clean_manager.start_download("whisper-large-v3")

    q = clean_manager.subscribe_events("whisper-large-v3")
    clean_manager.unsubscribe_events("whisper-large-v3", q)

    queues = clean_manager._queues.get("whisper-large-v3", [])
    assert q not in queues


def test_queue_bounded(clean_manager, clean_consent):
    """Queues are bounded — put_nowait drops events when full rather than
    growing unboundedly."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    with patch.object(
        dlmod, "_download_with_progress", return_value=(Path("/fake/path"), threading.Thread())
    ):
        clean_manager.start_download("whisper-large-v3")

    q = clean_manager.subscribe_events("whisper-large-v3")
    # Fill the queue.
    for i in range(200):
        with contextlib.suppress(Exception):
            q.put_nowait({"type": "test", "n": i})

    # Queue should not exceed its maxsize (100).
    assert q.qsize() <= 100


# -- SSE wire format ---------------------------------------------------------
#
# These exist because the download endpoints shipped with a literal
# backslash-n escape instead of a real newline as the frame terminator. An SSE
# event is terminated by a BLANK LINE, so every frame was malformed and a
# browser EventSource would have received nothing at all, forever, while the
# server reported perfectly healthy.
#
# None of the other tests caught it: they assert on decoded JSON payloads or on
# manager state, never on the bytes that actually go over the wire. The
# pre-existing summarize/cleanup endpoints in the same module were always
# correct, so this was a silent divergence from a working pattern sitting a few
# hundred lines above it.


async def test_sse_error_frame_is_terminated_by_a_blank_line():
    """An SSE frame must end with two real newlines, not an escape sequence."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3/download/progress")

    body = resp.text
    assert body.endswith("\n\n"), (
        f"SSE frame is not terminated by a blank line; body ends with {body[-8:]!r}"
    )
    assert "\\n" not in body, "SSE frame contains a literal backslash-n escape instead of a newline"


async def test_sse_error_frame_interpolates_the_model_key():
    """The error payload must name the actual model, not a literal placeholder."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3/download/progress")

    assert "{key}" not in resp.text, "error message contains an uninterpolated placeholder"
    assert "whisper-large-v3" in resp.text

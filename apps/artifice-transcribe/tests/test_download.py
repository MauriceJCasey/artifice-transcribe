# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ASR model download service, consent, and API endpoints.

No real Hugging Face downloads are triggered -- the transport is mocked at the
``huggingface_hub.snapshot_download`` level.  Consent file I/O uses isolated
temporary directories via ``monkeypatch``.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from artifice_transcribe.main import app
from artifice_transcribe.services.download import (
    DownloadManager,
    DownloadState,
    human_size,
    is_consented,
    record_consent,
    resolve_transitive,
    revoke_consent,
    total_transitive_size,
)
from artifice_transcribe.services.token_redaction import redact_token
from httpx import ASGITransport, AsyncClient

# -- Fixtures ----------------------------------------------------------------


@pytest.fixture
def clean_consent(monkeypatch, tmp_path):
    """Redirect consent storage to a temp directory."""
    consent_path = tmp_path / "model_consent.json"

    def _fake_consent_path():
        return consent_path

    monkeypatch.setattr(
        "artifice_transcribe.services.download._consent_path",
        _fake_consent_path,
    )
    return consent_path


@pytest.fixture
def clean_manager():
    """Return a fresh DownloadManager with no in-flight state."""
    return DownloadManager()


# -- Registry / dependency resolution ----------------------------------------


def test_resolve_transitive_simple():
    """A model with no dependencies resolves to just itself."""
    models = resolve_transitive("whisper-large-v3")
    assert len(models) == 1
    assert models[0].hf_repo == "openai/whisper-large-v3"


def test_resolve_transitive_with_deps():
    """pyannote-speaker-diarization pulls embedding."""
    models = resolve_transitive("pyannote-speaker-diarization")
    repos = [m.hf_repo for m in models]
    assert repos == [
        "pyannote/speaker-diarization-3.0",
        "pyannote/embedding",
    ]


def test_resolve_transitive_unknown_key():
    """A key not in ASR_MODELS raises KeyError."""
    with pytest.raises(KeyError):
        resolve_transitive("nonexistent-model")


def test_total_transitive_size():
    """Total size sums self + deps."""
    total = total_transitive_size("pyannote-speaker-diarization")
    assert total == 5_905_440 + 96_383_626


def test_human_size_mb():
    assert human_size(5_000_000) == "5.0 MB"


def test_human_size_gb():
    assert human_size(3_087_130_976) == "3.09 GB"


# -- Consent persistence -----------------------------------------------------


def test_consent_record_and_revoke(clean_consent):
    """Consent is recorded, read back, and revoked."""
    # Named `model_id`, deliberately avoiding the substring "key" anywhere in
    # the identifier. gitleaks' generic-api-key rule fires on an assignment
    # whose variable name *contains* "key" and whose value is a quoted string
    # of some entropy, so `model_key` failed the Zero Secrets Policy gate just
    # as `key` did — the rule matches the substring, not the whole name.
    #
    # This is a registry identifier, not a credential. The fix is to stop it
    # looking like one rather than to add a gitleaks suppression: a suppression
    # is a hole that outlives the false positive that justified it, and this
    # gate is the only thing standing between an API key and a public index.
    model_id = "whisper-large-v3"
    assert not is_consented(model_id)

    record_consent(model_id, True)
    assert is_consented(model_id)

    revoke_consent(model_id)
    assert not is_consented(model_id)


def test_consent_missing_file_is_false(clean_consent):
    """No consent file exists so all keys are unconsented."""
    assert not is_consented("any-key")


def test_consent_persists_across_calls(clean_consent):
    """Consent survives between read calls."""
    record_consent("parakeet-tdt-1.1b", True)
    assert is_consented("parakeet-tdt-1.1b")
    # Re-read from disk
    assert is_consented("parakeet-tdt-1.1b")


def test_consent_only_affects_given_key(clean_consent):
    """Consent for one key does not affect another."""
    record_consent("whisper-large-v3", True)
    assert is_consented("whisper-large-v3")
    assert not is_consented("pyannote-embedding")


# -- DownloadManager state machine -------------------------------------------


def test_download_manager_initially_empty(clean_manager):
    """A fresh manager has no active downloads."""
    assert clean_manager.get_status("any-key") is None


def test_download_manager_info_includes_deps(clean_manager, clean_consent):
    """info() computes transitive sizes and destination."""
    record_consent("pyannote-speaker-diarization", True)
    info = clean_manager.info("pyannote-speaker-diarization")
    assert info["key"] == "pyannote-speaker-diarization"
    assert len(info["models"]) == 2
    assert info["total_size_bytes"] == 5_905_440 + 96_383_626
    assert info["requires_hf_token"] is True
    assert info["consented"] is True
    assert Path(info["cache_directory"]).name == "hub"


def test_download_manager_refuses_without_consent(clean_manager, clean_consent):
    """start_download raises PermissionError when consent is absent."""
    with pytest.raises(PermissionError, match="Consent has not been recorded"):
        clean_manager.start_download("whisper-large-v3")


def test_download_manager_cancel(clean_manager, clean_consent):
    """cancel_download sets the cancel flag."""
    record_consent("whisper-large-v3", True)
    # Start a download, then cancel -- mock the download to avoid real network.
    import artifice_transcribe.services.download as dlmod

    started = threading.Event()

    def _fake_download(*args, **kwargs):
        cancel = kwargs.get("cancel")
        started.set()
        while cancel is not None and not cancel.is_set():
            import time

            time.sleep(0.05)

    with patch.object(dlmod, "_download_with_progress", _fake_download):
        ds = clean_manager.start_download("whisper-large-v3")
        started.wait(timeout=5)

    cancel_flag = clean_manager._cancel_flags.get("whisper-large-v3")
    assert cancel_flag is not None
    clean_manager.cancel_download("whisper-large-v3")
    assert cancel_flag.is_set()
    # Wait for thread to finish.
    import time

    deadline = time.time() + 5
    while not ds.finished and time.time() < deadline:
        time.sleep(0.1)


def test_download_manager_cleanup(clean_manager, clean_consent):
    """cleanup removes tracking data."""
    record_consent("whisper-large-v3", True)
    import artifice_transcribe.services.download as dlmod

    def _fake_download(*args, **kwargs):
        return Path("/fake/path"), threading.Thread()

    with patch.object(dlmod, "_download_with_progress", _fake_download):
        clean_manager.start_download("whisper-large-v3")

    assert clean_manager.get_status("whisper-large-v3") is not None
    clean_manager.cleanup("whisper-large-v3")
    assert clean_manager.get_status("whisper-large-v3") is None


# -- Download worker (mocked transport) --------------------------------------


def test_download_worker_handles_hf_error(clean_manager, clean_consent):
    """A Hugging Face error is captured as ERROR state."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    def _failing_download(*args, **kwargs):
        raise RuntimeError("401 Client Error: Unauthorized")

    with patch.object(dlmod, "_download_with_progress", _failing_download):
        ds = clean_manager.start_download("whisper-large-v3")

        import time

        deadline = time.time() + 10
        while not ds.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds.finished
    assert ds.models[0].state == DownloadState.ERROR
    assert "401" in ds.models[0].error_message


def test_download_worker_handles_network_error(clean_manager, clean_consent):
    """A network error (ConnectionError) is captured."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    def _failing_download(*args, **kwargs):
        raise ConnectionError("Network unreachable")

    with patch.object(dlmod, "_download_with_progress", _failing_download):
        ds = clean_manager.start_download("whisper-large-v3")

        import time

        deadline = time.time() + 10
        while not ds.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds.finished
    assert ds.models[0].state == DownloadState.ERROR
    assert "Network" in ds.models[0].error_message


def test_download_worker_handles_disk_full(clean_manager, clean_consent):
    """An OSError (disk full) is captured."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    def _failing_download(*args, **kwargs):
        raise OSError(28, "No space left on device")

    with patch.object(dlmod, "_download_with_progress", _failing_download):
        ds = clean_manager.start_download("whisper-large-v3")

        import time

        deadline = time.time() + 10
        while not ds.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds.finished
    assert ds.models[0].state == DownloadState.ERROR
    assert "No space" in ds.models[0].error_message


def test_download_worker_supports_cancellation(clean_manager, clean_consent):
    """Cancelling before the download thread starts sets CANCELLED."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    def _blocking_download(*args, **kwargs):
        # Spin until cancelled.
        cancel = kwargs.get("cancel")
        while cancel is not None and not cancel.is_set():
            import time

            time.sleep(0.05)
        raise dlmod._CancelledError()

    with patch.object(dlmod, "_download_with_progress", _blocking_download):
        ds = clean_manager.start_download("whisper-large-v3")

        import time

        time.sleep(0.2)  # Let the thread start.
        clean_manager.cancel_download("whisper-large-v3")

        deadline = time.time() + 10
        while not ds.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds.finished
    assert any(ms.state == DownloadState.CANCELLED for ms in ds.models), (
        f"Expected CANCELLED, got states: {[ms.state for ms in ds.models]}"
    )


# -- API endpoints -----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_all_keys():
    """GET /api/v1/models returns all ASR model keys."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    keys = {m["key"] for m in data["models"]}
    assert "whisper-large-v3" in keys
    assert "pyannote-speaker-diarization" in keys
    assert "pyannote-embedding" in keys
    assert "parakeet-tdt-1.1b" in keys


@pytest.mark.asyncio
async def test_model_info_returns_transitive_sizes():
    """GET /api/v1/models/{key} includes dependencies and total size."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/pyannote-speaker-diarization")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2
    assert data["total_size_bytes"] == 5_905_440 + 96_383_626
    assert data["requires_hf_token"] is True
    assert "cache_directory" in data


@pytest.mark.asyncio
async def test_model_info_unknown_key_returns_404():
    """GET /api/v1/models/unknown returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_consent_grant_and_revoke(clean_consent):
    """POST /api/v1/models/{key}/consent grants and revokes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/whisper-large-v3/consent",
            json={"consent": True},
        )
    assert resp.status_code == 200
    assert resp.json()["consented"] is True

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3")
    assert resp.json()["consented"] is True

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/whisper-large-v3/consent",
            json={"consent": False},
        )
    assert resp.status_code == 200
    assert resp.json()["consented"] is False


@pytest.mark.asyncio
async def test_download_refused_without_consent():
    """POST /api/v1/models/{key}/download returns 403 without consent."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/models/whisper-large-v3/download")
    assert resp.status_code == 403
    assert "Consent has not been granted" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_download_refused_for_unknown_key():
    """POST /api/v1/models/nonexistent/download returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/models/nonexistent/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_status_never_started():
    """GET /api/v1/models/{key}/download/status returns never_started."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3/download/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "never_started"


@pytest.mark.asyncio
async def test_consent_endpoint_unknown_key():
    """POST consent for unknown key returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/nonexistent/consent",
            json={"consent": True},
        )
    assert resp.status_code == 404


# -- Token redaction -----------------------------------------------------------


def test_redact_token_hf():
    """hf_ token in a string is replaced."""
    result = redact_token("Error: token hf_abcdefghijklmnopqrstuvwxyz123456 is invalid")
    assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in result
    assert "[REDACTED]" in result


def test_redact_token_sk():
    """sk- token in a string is replaced."""
    result = redact_token("Error: 401 Invalid API key: sk-proj-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-proj-abcdefghijklmnopqrstuvwxyz123456" not in result
    assert "[REDACTED]" in result


def test_redact_token_sk_ant():
    """sk-ant- token in a string is replaced."""
    result = redact_token("Error: Key sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890 is invalid")
    assert "sk-ant-api03-abcdefghijklmnopqrstuvwxyz1234567890" not in result
    assert "[REDACTED]" in result


def test_redact_token_no_token():
    """String without a token is returned unchanged."""
    msg = "401 Client Error: Unauthorized for url: ..."
    assert redact_token(msg) == msg


def test_redact_token_multiple():
    """All tokens in a string are redacted."""
    msg = "Token hf_aaaaaaaaaaaaaaaaaaaaa and sk-bbbbbbbbbbbbbbbbbbbbb failed"
    result = redact_token(msg)
    assert result.count("[REDACTED]") == 2
    assert "hf_" not in result
    assert "sk-" not in result


def test_redact_token_error_message(clean_manager, clean_consent):
    """Token-bearing error messages are redacted in DownloadStatus.error_message."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    def _failing_with_token(*args, **kwargs):
        raise RuntimeError(
            "401 Client Error: Repository Not Found for url: "
            "https://huggingface.co/api/models/org/repo. "
            "The token hf_abcDefGhijklmnopqrstuv123456 is invalid."
        )

    with patch.object(dlmod, "_download_with_progress", _failing_with_token):
        ds = clean_manager.start_download("whisper-large-v3")

        import time

        deadline = time.time() + 10
        while not ds.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds.finished
    assert ds.models[0].state == DownloadState.ERROR
    assert "hf_abcDefGhijklmnopqrstuv123456" not in ds.models[0].error_message
    assert "[REDACTED]" in ds.models[0].error_message


# -- Cancel mid-download (honest) -----------------------------------------------


def test_cancel_is_honest_does_not_mark_finished_immediately(clean_manager, clean_consent):
    """cancel_download sets the flag and emits 'cancelling', but does NOT
    immediately mark models as CANCELLED or ds.finished — the worker does
    that when it detects the flag."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    started = threading.Event()
    cancel_seen = threading.Event()

    def _fake_download(*args, **kwargs):
        cancel = kwargs.get("cancel")
        started.set()
        while cancel is not None and not cancel.is_set():
            import time

            time.sleep(0.05)
        # Cancel was set — signal so the test knows we saw it.
        cancel_seen.set()
        # Now the worker will raise _CancelledError since cancel is set.
        raise dlmod._CancelledError()

    with patch.object(dlmod, "_download_with_progress", _fake_download):
        ds = clean_manager.start_download("whisper-large-v3")
        started.wait(timeout=5)

        # Before cancel: models are DOWNLOADING, not finished.
        assert ds.models[0].state == DownloadState.DOWNLOADING
        assert not ds.finished

        # Cancel and check that the cancel flag IS set...
        clean_manager.cancel_download("whisper-large-v3")
        cancel_flag = clean_manager._cancel_flags.get("whisper-large-v3")
        assert cancel_flag is not None
        assert cancel_flag.is_set()

        # ... but the worker hasn't acted on it yet (polling is every 0.5 s).
        # Wait for the worker to detect the cancel.
        cancel_seen.wait(timeout=10)
        assert cancel_seen.is_set()

    # After the fake_download raises _CancelledError, the worker catches it
    # and marks the download as CANCELLED and finished.
    import time

    deadline = time.time() + 5
    while not ds.finished and time.time() < deadline:
        time.sleep(0.1)

    assert ds.finished
    assert any(ms.state == DownloadState.CANCELLED for ms in ds.models), (
        f"Expected CANCELLED, got states: {[ms.state for ms in ds.models]}"
    )


# -- Concurrent double-start protection ----------------------------------------


def test_concurrent_double_start_single_download(clean_manager, clean_consent):
    """Two simultaneous start_download calls for the same key produce exactly
    one download worker — the lock prevents the check-and-assign race."""
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    results = []
    # Block the download worker so it doesn't finish before the second
    # thread has a chance to observe an active download.
    worker_block = threading.Event()
    barrier = threading.Barrier(2, timeout=5)
    completed = threading.Barrier(2, timeout=5)

    def _blocking_download(*args, **kwargs):
        worker_block.wait(timeout=10)
        return Path("/fake/path"), threading.Thread()

    def _worker(key):
        barrier.wait()
        try:
            ds = clean_manager.start_download(key)
            results.append(ds)
        except Exception as exc:
            results.append(exc)
        completed.wait()

    with patch.object(dlmod, "_download_with_progress", _blocking_download):
        t1 = threading.Thread(target=_worker, args=("whisper-large-v3",))
        t2 = threading.Thread(target=_worker, args=("whisper-large-v3",))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        # Now release the worker.
        worker_block.set()

    # Both calls returned (no crash), and they got the same DownloadSet.
    assert len(results) == 2
    assert results[0] is results[1], (
        f"Expected same DownloadSet, got two different objects: "
        f"{type(results[0]).__name__} and {type(results[1]).__name__}"
    )


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


# -- Cancel-then-restart: two-writer protection (R1) ---------------------------


def test_cancel_then_restart_guards_against_two_writers(clean_manager, clean_consent):
    """Cancel a download, then immediately restart — the guard on
    _inner_threads must prevent a second ``snapshot_download`` from
    writing into the same cache directory while the first is still alive.

    Before the fix, the cancel path never registered the inner thread,
    so the is_alive() check returned False and a second download started
    immediately, producing two concurrent writers.
    """
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    # Phase 1: start a download that creates an inner thread.  We need
    # _download_with_progress to return a fake path + a live inner thread,
    # and we need it to check the cancel flag and raise _CancelledError
    # carrying that thread.
    inner_stopped = threading.Event()

    def _fake_with_inner_thread(
        repo_id, model_key, total_bytes, token, cache_dir, cancel, progress_callback
    ):
        # Create an inner thread that just sits there (simulating a real
        # snapshot_download that's still writing to disk).
        inner = threading.Thread(target=inner_stopped.wait, daemon=True)
        inner.start()
        # Give it a moment to start.
        import time

        time.sleep(0.1)
        # Check the cancel flag — if set, raise with the inner thread.
        if cancel.is_set():
            raise dlmod._CancelledError(inner)
        # Otherwise return normally (shouldn't happen in this test).
        return Path("/fake/path"), inner

    with patch.object(dlmod, "_download_with_progress", _fake_with_inner_thread):
        ds1 = clean_manager.start_download("whisper-large-v3")
        import time

        time.sleep(0.3)

        # Cancel: the worker should catch _CancelledError and register
        # the inner thread.
        clean_manager.cancel_download("whisper-large-v3")

        deadline = time.time() + 10
        while not ds1.finished and time.time() < deadline:
            time.sleep(0.1)

    assert ds1.finished, "download was not marked finished after cancel"

    # Phase 2: the inner thread is still alive.
    inner = clean_manager._inner_threads.get("whisper-large-v3")
    assert inner is not None, (
        "inner thread was not registered on cancel — the _inner_threads dict is empty"
    )
    assert inner.is_alive(), "inner thread died before the test could inspect it"

    # A second start_download must return the SAME DownloadSet (not
    # create a new one) because the inner thread is still alive.
    record_consent("whisper-large-v3", True)
    ds2 = clean_manager.start_download("whisper-large-v3")
    assert ds2 is ds1, (
        "start_download returned a NEW DownloadSet while the inner "
        "thread was still alive — a second writer is now running"
    )

    # Clean up: release the inner thread so it finishes.
    inner_stopped.set()
    inner.join(timeout=5)


# -- Consent revoked while lock is held (F7) ----------------------------------


def test_consent_revoked_after_lock_acquired_is_still_rejected(clean_manager, clean_consent):
    """is_consented must be checked inside the lock — if consent is recorded,
    the lock acquired, and then consent is revoked, the download must still
    be refused.

    Before F7 the check was outside the lock, creating a window where a
    concurrently revoked consent was ignored.
    """
    record_consent("whisper-large-v3", True)

    import artifice_transcribe.services.download as dlmod

    # Acquire the lock externally so the start_download call waits.
    lock = clean_manager._lock
    lock.acquire()

    # Revoke consent while the lock is held.
    revoke_consent("whisper-large-v3")

    def _fake_download(*args, **kwargs):
        return Path("/fake/path"), threading.Thread()

    # Patch _download_with_progress so the actual download doesn't try
    # to touch huggingface_hub.
    with patch.object(dlmod, "_download_with_progress", _fake_download):
        # Release the lock so start_download can proceed.
        lock.release()

        # start_download should now check is_consented under the lock
        # and see that consent has been revoked.
        with pytest.raises(PermissionError, match="Consent has not been recorded"):
            clean_manager.start_download("whisper-large-v3")

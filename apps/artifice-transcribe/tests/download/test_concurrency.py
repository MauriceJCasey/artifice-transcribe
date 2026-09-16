# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Concurrency and cancel/restart edge-case tests for the download service."""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from artifice_transcribe.services.download import (
    DownloadState,
    record_consent,
    revoke_consent,
)

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

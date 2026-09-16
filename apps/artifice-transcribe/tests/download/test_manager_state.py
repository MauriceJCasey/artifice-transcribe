# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DownloadManager state-machine and download-worker (mocked transport) tests."""

import threading
from pathlib import Path
from unittest.mock import patch

import pytest
from artifice_transcribe.services.download import (
    DownloadState,
    record_consent,
)

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

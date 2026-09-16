# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download worker helpers — real byte-level progress polling and cache counting."""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from ..token_redaction import redact_token

logger = logging.getLogger(__name__)


class _CancelledError(Exception):
    """Raised when the user cancels the download.

    Carries *inner_thread* — the daemon thread running ``snapshot_download`` —
    so the caller can register it for ``is_alive()`` checks even on the cancel
    path, where ``_download_with_progress`` raises before returning the
    thread normally.
    """

    def __init__(self, inner_thread: threading.Thread | None = None) -> None:
        self.inner_thread = inner_thread
        super().__init__("Download cancelled")


def _download_with_progress(
    repo_id: str,
    model_key: str,
    total_bytes: int,
    token: str | None,
    cache_dir: Path,
    cancel: threading.Event,
    progress_callback,
) -> tuple[Path, threading.Thread]:
    """Download model files from *repo_id* with progress monitoring.

    Uses ``huggingface_hub.snapshot_download`` for the actual download (it
    handles auth, caching, and resumption).  Progress is monitored by polling
    the ``.incomplete`` download files in the cache — ``snapshot_download``
    streams to temp files, and polling their sizes gives real byte progress
    rather than a 0→100 jump on completion.

    Returns ``(snapshot_path, inner_thread)`` on success.  The inner thread
    is the daemon thread that runs ``snapshot_download`` — it may outlive
    the polling loop after a cancel, and the caller stores it so
    ``start_download`` can check :meth:`~threading.Thread.is_alive` before
    starting a new download into the same cache directory.

    Raises ``_CancelledError`` when the ``cancel`` event is set.
    Raises ``RuntimeError`` with a redacted message on failure.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.constants import REPO_ID_SEPARATOR

    cache_dir.mkdir(parents=True, exist_ok=True)

    repo_folder = repo_id.replace("/", REPO_ID_SEPARATOR)

    started = threading.Event()

    # Launch the download in a sub-thread so we can monitor progress from here.
    download_result: list[Exception | Path] = []
    download_done = threading.Event()

    def _download() -> None:
        started.set()
        try:
            result = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(cache_dir),
                token=token,
                resume_download=True,
            )
            download_result.append(Path(result))
        except Exception as exc:
            download_result.append(exc)
        finally:
            download_done.set()

    dl_thread = threading.Thread(target=_download, daemon=True)
    dl_thread.start()

    if not started.wait(timeout=10):
        logger.warning(
            "Download thread for %s did not start within 10 s — may be stuck",
            repo_id,
        )

    # Poll the snapshot directory for file sizes.
    last_reported = 0
    while not download_done.is_set():
        if cancel.is_set():
            # The inner snapshot_download has no cancel hook and will keep
            # running, but we stop polling and raise so the worker cleans up.
            raise _CancelledError(dl_thread)

        # Count bytes in the snapshot directory and blobs.
        current = _count_cache_bytes(cache_dir, repo_folder)
        if current > last_reported:
            last_reported = current
            progress_callback(
                min(current / max(total_bytes, 1), 1.0),
                min(current, total_bytes),
            )

        download_done.wait(timeout=0.5)

    # Final check — capture the result.
    dl_thread.join(timeout=5)

    if download_result:
        result = download_result[0]
        if isinstance(result, Exception):
            msg = redact_token(str(result))
            # Do *not* chain the raw exception as __cause__ — it may carry an
            # unredacted token in its message or args.
            raise RuntimeError(f"Download failed for {repo_id}: {msg}")
        return result, dl_thread

    # If we got here, something unexpected happened.
    raise RuntimeError(f"Download for {repo_id} did not complete")


def _count_cache_bytes(cache_dir: Path, repo_folder: str) -> int:
    """Count bytes currently on disk for a model in the HF cache.

    Walks the ``models--{repo_folder}`` tree and sums file sizes,
    deduplicating by inode so hardlinks (snapshots → blobs) are not
    double-counted.
    """
    total = 0
    base = cache_dir / f"models--{repo_folder}"
    if not base.exists():
        return 0
    seen_inodes: set[tuple[int, int]] = set()
    for p in base.rglob("*"):
        if p.is_file():
            try:
                st = p.stat()
                ino = (st.st_dev, st.st_ino)
                if ino not in seen_inodes:
                    seen_inodes.add(ino)
                    total += st.st_size
            except OSError:
                pass
    return total

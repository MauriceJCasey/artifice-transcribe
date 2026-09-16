# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""DownloadManager — orchestrates in-flight ASR model downloads with progress."""

from __future__ import annotations

import contextlib
import threading
from queue import Full, Queue
from typing import Any

from model_harness.registry import AsrModelInfo

# The worker helpers that are monkeypatched by name in tests
# (`patch.object(dlmod, "_download_with_progress", ...)` and
# `dlmod._CancelledError()`), where ``dlmod`` is ``artifice_transcribe.services.download``.
# They are referenced as *module attributes* of this package (a dynamic lookup at
# call time) rather than bound at import time, so a patch on the package reaches
# the manager's call sites. This mirrors the ``pdf_export`` facade in the OCR app.
import artifice_transcribe.services.download as _dl

from ..token_redaction import redact_token
from . import is_consented
from .models import DownloadSet, DownloadState, DownloadStatus
from .registry import find_registry_key, hf_cache_dir, human_size, resolve_transitive

# ── Download manager ─────────────────────────────────────────────────────────

_SSE_QUEUE_DEPTH = 100
"""Maximum events buffered per SSE client before older events are dropped."""


class DownloadManager:
    """Manages in-flight ASR model downloads with real progress reporting.

    A single instance tracks all active downloads.  Progress is published via
    SSE events through per-client queues so each viewer receives every event.

    The download itself uses ``huggingface_hub.snapshot_download`` for caching,
    auth, and resumption.  Progress is measured by monitoring the ``.incomplete``
    file that ``snapshot_download`` writes during transfer — this gives real
    byte-level progress, not a 0→100 jump.
    """

    def __init__(self) -> None:
        self._active: dict[str, DownloadSet] = {}
        """key → active DownloadSet."""

        self._cancel_flags: dict[str, threading.Event] = {}
        """key → cancel event for the download worker."""

        self._inner_threads: dict[str, threading.Thread] = {}
        """key → inner ``snapshot_download`` thread (may outlive the worker)."""

        # Per-SSE-client queues — one new queue per ``subscribe_events`` call.
        self._queues: dict[str, list[Queue[dict[str, Any]]]] = {}
        """key → list of per-client event queues for SSE streaming."""

        self._lock = threading.Lock()
        """Guards check-and-create in :meth:`start_download`."""

    # ── Progress events ──────────────────────────────────────────────────

    def _emit(self, key: str, event: dict[str, Any]) -> None:
        """Push a progress event to every registered SSE queue (non-blocking)."""
        for q in self._queues.get(key, ()):
            # A client that is not draining its queue must never block the
            # download worker, so a full queue drops the event rather than
            # waiting. Progress events are advisory; the terminal `completed`
            # or `error` state is also recorded on the DownloadSet, so a
            # dropped event cannot lose the outcome — only intermediate ticks.
            with contextlib.suppress(Full):
                q.put_nowait(event)

    # ── Public API ────────────────────────────────────────────────────────

    def info(self, key: str) -> dict[str, Any]:
        """Return model info for a consent dialog: name, transitive size, token
        required, and the resolved on-disk destination."""
        models = resolve_transitive(key)
        total = sum(m.size_bytes for m in models)
        need_token = any(m.requires_hf_token for m in models)
        cache = hf_cache_dir()

        return {
            "key": key,
            "models": [
                {
                    "key": key if m is models[0] else find_registry_key(m),
                    "hf_repo": m.hf_repo,
                    "size_bytes": m.size_bytes,
                    "requires_hf_token": m.requires_hf_token,
                    "description": m.description,
                }
                for m in models
            ],
            "total_size_bytes": total,
            "total_size_human": human_size(total),
            "requires_hf_token": need_token,
            "cache_directory": str(cache),
            "consented": is_consented(key),
        }

    def start_download(self, key: str, token: str = "") -> DownloadSet:
        """Begin downloading *key* and its dependencies.

        The download runs in a background thread.  Progress events are pushed
        to per-client queues; call :meth:`subscribe_events` to stream them.

        Requires that consent has been recorded for *key*.  Raises
        ``PermissionError`` if not.

        If *token* is ``""`` and any model in the transitive set requires
        authentication, the download will fail with an informative
        ``error_message`` — gated repos return HTTP 401 from Hugging Face.

        This method is guarded by a lock so that two concurrent calls see a
        consistent picture of the active set — no two observers can race into
        creating duplicate download workers for the same key.
        """
        with self._lock:
            if not is_consented(key):
                raise PermissionError(
                    f"Consent has not been recorded for '{key}'. "
                    f"Call POST /api/v1/models/{key}/consent first."
                )

            existing = self._active.get(key)
            if existing is not None:
                # Still actively downloading (not in a terminal state).
                if not existing.finished:
                    return existing
                # Terminal state, but is the inner snapshot_download thread
                # still alive?  If so, the previous download hasn't fully
                # stopped yet — treat it as still active to avoid a second
                # ``snapshot_download`` writing into the same cache directory.
                inner = self._inner_threads.get(key)
                if inner is not None and inner.is_alive():
                    return existing
                # Previous download is truly done — allow restart.

            models = resolve_transitive(key)
            ds = DownloadSet(request_key=key)
            ds.models = [
                DownloadStatus(
                    key=find_registry_key(m),
                    hf_repo=m.hf_repo,
                    total_bytes=m.size_bytes,
                    cache_path=str(hf_cache_dir()),
                )
                for m in models
            ]

            self._active[key] = ds
            self._cancel_flags[key] = threading.Event()
            self._queues.setdefault(key, [])

            thread = threading.Thread(
                target=self._download_worker,
                args=(key, models, token),
                daemon=True,
            )
            thread.start()
            return ds

    def cancel_download(self, key: str) -> None:
        """Request cancellation of an in-flight download.

        Sets the cancel flag so the download polling loop stops on its next
        iteration.  Does **not** immediately mark the download as cancelled —
        the worker thread handles that once it observes the flag (within one
        polling interval).

        A ``cancelling`` event is emitted so the UI can show honest status
        (the transfer may continue for a few moments until the polling loop
        detects the flag).
        """
        cancel = self._cancel_flags.get(key)
        if cancel is not None:
            cancel.set()
            self._emit(key, {"type": "cancelling", "key": key})

    def subscribe_events(self, key: str) -> Queue[dict[str, Any]]:
        """Create a new per-client event queue for *key* and return it.

        Each caller gets its own queue — multiple SSE viewers do not compete
        for events.  The queue is bounded so a disconnected client does not
        cause unbounded growth.

        Returns an empty queue even if *key* is unknown — the caller should
        check :meth:`get_status` first.
        """
        q: Queue[dict[str, Any]] = Queue(maxsize=_SSE_QUEUE_DEPTH)
        self._queues.setdefault(key, []).append(q)
        return q

    def unsubscribe_events(self, key: str, queue: Queue[dict[str, Any]]) -> None:
        """Remove a per-client event queue registered for *key*."""
        queues = self._queues.get(key, [])
        with contextlib.suppress(ValueError):
            queues.remove(queue)

    def get_status(self, key: str) -> DownloadSet | None:
        """Return the current :class:`DownloadSet` for *key*, or ``None``."""
        return self._active.get(key)

    def cleanup(self, key: str) -> None:
        """Remove tracking for a finished download.

        Does NOT delete downloaded files from disk.
        """
        with self._lock:
            self._active.pop(key, None)
            self._queues.pop(key, None)
            self._cancel_flags.pop(key, None)
            self._inner_threads.pop(key, None)

    # ── Worker ────────────────────────────────────────────────────────────

    def _download_worker(
        self,
        request_key: str,
        models: list[AsrModelInfo],
        token: str,
    ) -> None:
        """Background thread: download each model in sequence.

        Each model file is downloaded via ``huggingface_hub.snapshot_download``,
        which handles caching, authentication, and resumption.  Progress is
        read from the ``.incomplete`` file that the library writes during
        transfer.
        """
        ds = self._active[request_key]
        cancel = self._cancel_flags[request_key]
        cache_dir = hf_cache_dir()
        success_count = 0
        error_occurred = False

        for i, info in enumerate(models):
            if cancel.is_set():
                ds.models[i].state = DownloadState.CANCELLED
                ds.finished = True
                ds.error_message = "Cancelled by user"
                self._emit(
                    request_key,
                    {"type": "cancelled", "key": request_key},
                )
                return

            ms = ds.models[i]
            ms.state = DownloadState.DOWNLOADING
            ms.downloaded_bytes = 0
            ds.started = True

            self._emit(
                request_key,
                {
                    "type": "progress",
                    "key": request_key,
                    "model_idx": i,
                    "model_key": ms.key,
                    "model_total": len(models),
                    "hf_repo": info.hf_repo,
                    "total_bytes": ms.total_bytes,
                    "downloaded_bytes": 0,
                    "state": "downloading",
                },
            )

            try:
                downloaded_path, inner_thread = _dl._download_with_progress(
                    repo_id=info.hf_repo,
                    model_key=ms.key,
                    total_bytes=ms.total_bytes,
                    token=token if info.requires_hf_token else None,
                    cache_dir=cache_dir,
                    cancel=cancel,
                    # `i`, `ms` and `info` are bound as default arguments, not
                    # captured by reference.  This is defensive (satisfies ruff
                    # B023): ``_download_with_progress`` returns before the
                    # outer loop advances, so a late callback cannot fire in
                    # practice, but binding by value costs nothing and prevents
                    # a stray delayed callback from reporting the wrong model.
                    progress_callback=lambda pct, downloaded, _idx=i, _ms=ms, _info=info: (
                        self._emit(
                            request_key,
                            {
                                "type": "progress",
                                "key": request_key,
                                "model_idx": _idx,
                                "model_key": _ms.key,
                                "model_total": len(models),
                                "hf_repo": _info.hf_repo,
                                "total_bytes": _ms.total_bytes,
                                "downloaded_bytes": downloaded,
                                "state": "downloading",
                            },
                        )
                    ),
                )
                # Track the inner snapshot_download thread so start_download
                # can check is_alive() before starting a new download.
                # Registered under the lock because start_download reads it
                # under the lock — a dict write in one thread that is read in
                # another without synchronisation is a data race.
                if inner_thread is not None:
                    with self._lock:
                        self._inner_threads[request_key] = inner_thread

                ms.downloaded_bytes = ms.total_bytes
                ms.state = DownloadState.DONE
                self._emit(
                    request_key,
                    {
                        "type": "progress",
                        "key": request_key,
                        "model_idx": i,
                        "model_key": ms.key,
                        "model_total": len(models),
                        "hf_repo": info.hf_repo,
                        "total_bytes": ms.total_bytes,
                        "downloaded_bytes": ms.total_bytes,
                        "state": "done",
                        "cache_path": str(downloaded_path),
                    },
                )
                success_count += 1

            except _dl._CancelledError as exc:
                # Register the inner thread so start_download's is_alive()
                # check can prevent a second writer into the same cache dir.
                if exc.inner_thread is not None:
                    with self._lock:
                        self._inner_threads[request_key] = exc.inner_thread
                ms.state = DownloadState.CANCELLED
                ds.finished = True
                ds.error_message = "Cancelled by user"
                self._emit(
                    request_key,
                    {"type": "cancelled", "key": request_key},
                )
                return

            except Exception as exc:
                ms.state = DownloadState.ERROR
                # Redact any token-bearing error messages at the source.
                ms.error_message = redact_token(str(exc))
                error_occurred = True
                self._emit(
                    request_key,
                    {
                        "type": "error",
                        "key": request_key,
                        "model_idx": i,
                        "model_key": ms.key,
                        "error": ms.error_message,
                    },
                )
                # Continue with remaining models unless cancelled.
                if cancel.is_set():
                    ds.finished = True
                    ds.error_message = "Cancelled by user"
                    return

        ds.finished = True
        if error_occurred:
            ds.error_message = "One or more models failed to download"
        elif success_count > 0:
            self._emit(
                request_key,
                {
                    "type": "completed",
                    "key": request_key,
                    "total_models": len(models),
                    "success_count": success_count,
                },
            )


# ── Module-level singleton ───────────────────────────────────────────────────

_download_manager: DownloadManager | None = None


def get_download_manager() -> DownloadManager:
    """Return the module-level :class:`DownloadManager` singleton."""
    global _download_manager
    if _download_manager is None:
        _download_manager = DownloadManager()
    return _download_manager

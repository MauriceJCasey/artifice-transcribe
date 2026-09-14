# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ASR model download service — consent, size disclosure, and progress.

This module manages model-weight downloads from Hugging Face with explicit
user consent, real byte-level progress reporting, and accurate transitive-size
disclosure.  It does NOT import ``torch`` at module scope — the lightweight
install (no ``--extra asr``) must be able to serve the model-list and consent
endpoints, and this module is the surface that tells a bare install *what* is
available before it downloads anything.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from queue import Full, Queue
from typing import Any

from model_harness.registry import ASR_MODELS, AsrModelInfo

from .token_redaction import redact_token

logger = logging.getLogger(__name__)


# ── Cache directory (matches huggingface_hub's default) ──────────────────────


def hf_cache_dir() -> Path:
    """Return the Hugging Face cache directory (platform-aware)."""
    import platformdirs

    default = Path(platformdirs.user_cache_dir("huggingface", "huggingface")) / "hub"
    env = os.environ.get("HF_HUB_CACHE", "")
    return Path(env) if env else default


# ── Transitive dependency resolution ─────────────────────────────────────────


def resolve_transitive(key: str) -> list[AsrModelInfo]:
    """Return the full ordered set of models needed for *key*.

    The first entry is always the requested model itself; any dependencies
    follow in the order they are discovered.  Raises ``KeyError`` if *key*
    is not in :data:`~model_harness.registry.ASR_MODELS`.
    """
    info = ASR_MODELS[key]
    seen: set[str] = {key}
    result: list[AsrModelInfo] = [info]

    # Breadth-first so a direct dependency always appears before its own deps.
    queue: list[str] = list(info.depends_on)
    while queue:
        dep_key = queue.pop(0)
        if dep_key in seen:
            continue
        seen.add(dep_key)
        dep_info = ASR_MODELS[dep_key]
        result.append(dep_info)
        queue.extend(d for d in dep_info.depends_on if d not in seen)

    return result


def total_transitive_size(key: str) -> int:
    """Sum of all model weights for *key* and its dependencies."""
    return sum(info.size_bytes for info in resolve_transitive(key))


def requires_token(key: str) -> bool:
    """``True`` if any model in the transitive set needs an HF token."""
    return any(info.requires_hf_token for info in resolve_transitive(key))


# ── Consent persistence ──────────────────────────────────────────────────────


def _consent_path() -> Path:
    """Return the per-user consent file path (``platformdirs``, not CWD)."""
    import platformdirs

    data_dir = Path(platformdirs.user_data_dir("artifice-transcribe", "ArtificeSuite"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "model_consent.json"


def _load_consents() -> dict[str, bool]:
    """Return ``{model_key: True}`` for every consented model."""
    path = _consent_path()
    if not path.exists():
        return {}
    try:
        from secure_io import ensure_restricted

        ensure_restricted(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read consent file at %s — treating as empty", path)
        return {}


def _save_consents(data: dict[str, bool]) -> None:
    """Persist consent decisions with OS-appropriate access controls."""
    from secure_io import write_private_json

    write_private_json(_consent_path(), data)


def is_consented(key: str) -> bool:
    """Return ``True`` if the user has recorded consent for *key*."""
    return bool(_load_consents().get(key, False))


def record_consent(key: str, consented: bool = True) -> None:
    """Set consent for *key* to *consented* and persist."""
    data = _load_consents()
    if consented:
        data[key] = True
    else:
        data.pop(key, None)
    _save_consents(data)


def revoke_consent(key: str) -> None:
    """Remove consent for *key*."""
    record_consent(key, consented=False)


# ── Download state machine ───────────────────────────────────────────────────


class DownloadState(Enum):
    """States for an in-flight model download."""

    IDLE = "idle"
    """Not started."""
    DOWNLOADING = "downloading"
    """Actively transferring bytes."""
    VERIFYING = "verifying"
    """Download complete, checksum verification in progress."""
    DONE = "done"
    """Successfully downloaded and verified."""
    ERROR = "error"
    """Download failed — see ``error_message``."""
    CANCELLED = "cancelled"
    """User cancelled before completion."""


@dataclass
class DownloadStatus:
    """Snapshot of a single model's download progress."""

    key: str
    """Registry key of the model being downloaded."""
    state: DownloadState = DownloadState.IDLE
    """Current state of this download."""
    hf_repo: str = ""
    """Hugging Face repository being downloaded."""
    total_bytes: int = 0
    """Expected total bytes for this model."""
    downloaded_bytes: int = 0
    """Bytes transferred so far."""
    error_message: str = ""
    """Human-readable error if state is ``ERROR``."""
    cache_path: str = ""
    """Resolved cache directory where files land."""


@dataclass
class DownloadSet:
    """Overall status for a model-and-dependencies download job."""

    request_key: str
    """The original model key the user requested."""
    models: list[DownloadStatus] = field(default_factory=list)
    """One status entry per model in the transitive set (self first, then deps)."""
    started: bool = False
    """``True`` once the first byte transfer begins."""
    finished: bool = False
    """``True`` when every model has reached a terminal state."""
    error_message: str = ""
    """Overall error if the download set failed."""


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
                downloaded_path, inner_thread = _download_with_progress(
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

            except _CancelledError as exc:
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


# ── Internal helpers ─────────────────────────────────────────────────────────


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


def find_registry_key(info: AsrModelInfo) -> str:
    """Reverse-lookup the registry key for an :class:`AsrModelInfo` instance."""
    for k, v in ASR_MODELS.items():
        if v is info:
            return k
    return info.hf_repo


def human_size(num_bytes: int) -> str:
    """Return a human-readable size string (MB or GB)."""
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    return f"{num_bytes / 1_000_000:.1f} MB"


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


# ── Module-level singleton ───────────────────────────────────────────────────

_download_manager: DownloadManager | None = None


def get_download_manager() -> DownloadManager:
    """Return the module-level :class:`DownloadManager` singleton."""
    global _download_manager
    if _download_manager is None:
        _download_manager = DownloadManager()
    return _download_manager

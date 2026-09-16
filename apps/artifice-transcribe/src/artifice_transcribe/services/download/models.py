# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download state machine — enums and status dataclasses for in-flight downloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The ASR backend contract.

This module holds the data types and interface every speech-to-text engine
must satisfy.  It deliberately contains no implementation: WhisperX is the
only current backend (see :class:`artifice_transcribe.services.transcription.
WhisperXEngine`), but a future NVIDIA Parakeet / NeMo engine would implement
the same :class:`ASRBackend` protocol and slot in without the rest of the app
knowing which engine it is talking to.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    segments: list[Segment]
    speaker_embeddings: dict[str, list[float]]


@runtime_checkable
class ASRBackend(Protocol):
    """The interface every transcription engine exposes to its callers.

    ``@runtime_checkable`` lets a conformance test assert
    ``isinstance(engine, ASRBackend)``, but the protocol is otherwise purely
    structural — a backend may satisfy it by duck-typing without subclassing.
    """

    def transcribe(
        self,
        audio_path: str | Path,
        *,
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        progress_callback: Callable[[float], None] | None = None,
        custom_vocabulary: str | None = None,
        hotwords: str | None = None,
    ) -> TranscriptionResult: ...

    def unload(self) -> None: ...

    def health_check(self) -> dict: ...

    def preload(self) -> dict: ...

    def extract_speaker_embedding(self, audio_path: str | Path) -> np.ndarray: ...

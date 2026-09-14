# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Conformance tests for the ASR backend contract.

:class:`artifice_transcribe.services.asr_backend.ASRBackend` is the structural
interface every speech-to-text engine must satisfy.  These tests pin that
contract so a future engine (NVIDIA Parakeet via NeMo) can slot in behind the
same shape, and so the current :class:`WhisperXEngine` cannot drift from it.

The ``@runtime_checkable`` ``isinstance`` checks exercise the protocol's
structural membership test — not a nominal subclass relationship — which is
exactly the property the interface exists to provide.
"""

from __future__ import annotations

import importlib.util

import pytest
from artifice_transcribe.services.asr_backend import ASRBackend, Segment, TranscriptionResult

_ASR_AVAILABLE = importlib.util.find_spec("torch") is not None


def test_segment_and_result_live_on_the_contract():
    """The data contract is engine-agnostic and importable without the ASR stack."""
    seg = Segment(speaker="SPEAKER_00", start=0.0, end=1.5, text="hello")
    result = TranscriptionResult(segments=[seg], speaker_embeddings={"SPEAKER_00": [0.1]})
    assert result.segments[0].text == "hello"
    assert result.speaker_embeddings["SPEAKER_00"] == [0.1]


def _full_backend():
    """A minimal, dependency-free class exposing every ASRBackend method."""

    class FullBackend:
        def transcribe(self, audio_path, **kwargs):
            return TranscriptionResult(segments=[], speaker_embeddings={})

        def unload(self) -> None: ...

        def health_check(self) -> dict:
            return {}

        def preload(self) -> dict:
            return {}

        def extract_speaker_embedding(self, audio_path):
            return [0.0]

    return FullBackend()


def _partial_backend():
    """Only ``transcribe`` + ``unload`` — the surface a worker-only stand-in
    (see ``test_api_e2e.FakeEngine``) implements."""

    class PartialBackend:
        def transcribe(self, audio_path, **kwargs):
            return TranscriptionResult(segments=[], speaker_embeddings={})

        def unload(self) -> None: ...

    return PartialBackend()


def test_full_backend_satisfies_protocol():
    """A class implementing all five methods is an ASRBackend by structure."""
    assert isinstance(_full_backend(), ASRBackend)


def test_partial_backend_does_not_satisfy_protocol():
    """A class missing the health/preload/embedding methods is rejected.

    This proves ``@runtime_checkable`` is doing real structural membership
    testing rather than a nominal subclass check: if it were nominal, a
    duck-type with only ``transcribe``/``unload`` could slip through.
    """
    assert not isinstance(_partial_backend(), ASRBackend)


@pytest.mark.skipif(not _ASR_AVAILABLE, reason="ASR stack (torch/whisperx) not installed")
def test_whisperx_engine_satisfies_asr_backend():
    """The concrete WhisperX engine is a member of the ASRBackend protocol."""
    from artifice_transcribe.services.transcription import WhisperXEngine

    # Construction only records settings — no model is loaded until first use.
    # ``device="cpu"`` avoids the CUDA probe in ``_resolve_device``.
    engine = WhisperXEngine(model_size="base", device="cpu", hf_token="", diarization_model="")
    assert isinstance(engine, ASRBackend)

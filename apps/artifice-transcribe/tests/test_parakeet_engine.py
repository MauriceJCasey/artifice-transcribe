# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Unit tests for the Parakeet (NeMo) ASR backend.

NeMo is not installed in this dev environment or in CI, so these tests are
structured the way ``test_silence.py`` was: everything that needs the real
package is either skipped (via ``importlib.util.find_spec``) or exercised
against a stdlib stand-in.  The segment-building and speaker-assignment logic
is tested end-to-end with ``nemo`` / ``torch`` / ``whisperx`` faked into
``sys.modules``, so the real code path runs with no heavy dependency present.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections import namedtuple

import pytest
from artifice_transcribe.services.asr_backend import ASRBackend
from artifice_transcribe.services.parakeet_engine import (
    DEFAULT_MODEL_NAME,
    ParakeetEngine,
    ParakeetRequiresCuda,
    _assign_speaker_by_overlap,
)

_NEMO_AVAILABLE = importlib.util.find_spec("nemo") is not None

# Mirrors whisperx.DiarizationPipeline's returned DataFrame columns:
# `itertuples(index=False)` yields namedtuples with these fields.
Turn = namedtuple("Turn", ["segment", "label", "speaker", "start", "end"])


# ── Fake dependency stack ─────────────────────────────────────────────────


def _fake_torch(*, cuda_available: bool) -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda_available)
    return torch


def _fake_nemo(asr_model) -> types.ModuleType:
    """A ``nemo.collections.asr.models.ASRModel`` whose ``from_pretrained``
    returns *asr_model*."""
    nemo = types.ModuleType("nemo")
    collections = types.ModuleType("nemo.collections")
    asr = types.ModuleType("nemo.collections.asr")
    models = types.ModuleType("nemo.collections.asr.models")

    class ASRModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return asr_model

    models.ASRModel = ASRModel
    asr.models = models
    collections.asr = asr
    nemo.collections = collections
    return nemo


def _fake_whisperx(diarize_pipeline_factory) -> types.ModuleType:
    whisperx = types.ModuleType("whisperx")
    diarize = types.ModuleType("whisperx.diarize")
    diarize.DiarizationPipeline = diarize_pipeline_factory
    whisperx.diarize = diarize
    return whisperx


class FakeHypothesis:
    def __init__(self, timestamp: dict):
        self.timestamp = timestamp


class FakeASRModel:
    """A stand-in for NeMo's loaded model: ``transcribe([...], timestamps=True)``
    returns a list of hypotheses whose ``.timestamp`` matches the documented
    ``{"segment": [{"segment": text, "start": float, "end": float}, ...]}``
    shape (verified against nemo.collections.asr.parts.utils.timestamp_utils)."""

    def __init__(self, hypotheses):
        self._hypotheses = hypotheses
        self.transcribe_calls = []

    def transcribe(self, audio_paths, *, timestamps=False):
        self.transcribe_calls.append((list(audio_paths), timestamps))
        return self._hypotheses


class FakeDiarizeDF:
    """A minimal stand-in for the pandas DataFrame DiarizationPipeline returns."""

    def __init__(self, turns: list[tuple[float, float, str]]):
        self._turns = turns

    def itertuples(self, index=False):
        return [Turn(None, None, speaker, start, end) for start, end, speaker in self._turns]


class FakeDiarizePipeline:
    def __init__(self, turns, embeddings, *, model_name=None, token=None, device=None):
        self._turns = turns
        self._embeddings = embeddings

    def __call__(
        self, audio_path, *, min_speakers=None, max_speakers=None, return_embeddings=False
    ):
        return FakeDiarizeDF(self._turns), self._embeddings if return_embeddings else None


def _install_stack(monkeypatch, *, asr_model, diarize_turns, embeddings):
    """Inject fake ``torch`` / ``nemo`` / ``whisperx`` into ``sys.modules``."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=True))
    nemo_mod = _fake_nemo(asr_model)
    monkeypatch.setitem(sys.modules, "nemo", nemo_mod)
    monkeypatch.setitem(sys.modules, "nemo.collections", nemo_mod.collections)
    monkeypatch.setitem(sys.modules, "nemo.collections.asr", nemo_mod.collections.asr)
    whisperx_mod = _fake_whisperx(lambda **kw: FakeDiarizePipeline(diarize_turns, embeddings, **kw))
    monkeypatch.setitem(sys.modules, "whisperx", whisperx_mod)
    monkeypatch.setitem(sys.modules, "whisperx.diarize", whisperx_mod.diarize)


# ── ParakeetRequiresCuda ───────────────────────────────────────────────────


def test_parakeet_requires_cuda_is_a_clean_exception():
    exc = ParakeetRequiresCuda()
    # public_message must be a string literal, not derived from a wrapped
    # exception's str() (CodeQL taint-tracker constraint — see AsrUnavailable).
    assert "CUDA" in exc.public_message
    assert "WhisperX" in exc.public_message
    assert str(exc) == exc.public_message


def test_engine_constructs_without_loading_models():
    """Construction records settings only — no CUDA probe, no NeMo import."""
    engine = ParakeetEngine()
    assert engine._asr_model is None
    assert engine._diarize_model is None
    assert engine._device == "cuda"  # "auto" resolves to cuda (CUDA-only)


# ── ASRBackend conformance ─────────────────────────────────────────────────


def test_parakeet_engine_satisfies_asr_backend():
    """ParakeetEngine is a member of the ASRBackend protocol by structure.

    Unlike WhisperXEngine, importing ParakeetEngine needs no ASR stack (its
    heavy imports are lazy), so this runs without NeMo installed.
    """
    engine = ParakeetEngine()
    assert isinstance(engine, ASRBackend)


# ── Speaker-assignment algorithm (pure, dependency-free) ───────────────────


def test_overlap_assignment_falls_back_when_no_turn_overlaps():
    assert _assign_speaker_by_overlap(0.0, 1.0, []) == "SPEAKER_00"
    assert _assign_speaker_by_overlap(5.0, 6.0, [(0.0, 1.0, "A")]) == "SPEAKER_00"


def test_overlap_assignment_picks_the_full_overlap_speaker():
    turns = [(0.0, 1.0, "A"), (1.0, 2.0, "B")]
    assert _assign_speaker_by_overlap(0.0, 1.0, turns) == "A"
    assert _assign_speaker_by_overlap(1.0, 2.0, turns) == "B"


def test_overlap_assignment_picks_greatest_overlap():
    turns = [(0.0, 0.6, "A"), (0.4, 1.0, "B")]  # segment 0.0-1.0: A=0.6, B=0.6
    # Overlap A = 0.6, B = 0.6 -> tie, earliest (A) wins.
    assert _assign_speaker_by_overlap(0.0, 1.0, turns) == "A"


def test_overlap_assignment_greatest_overlap_not_first():
    turns = [(0.0, 0.3, "A"), (0.3, 1.0, "B")]
    # A overlaps 0.3, B overlaps 0.7 -> B wins.
    assert _assign_speaker_by_overlap(0.0, 1.0, turns) == "B"


# ── transcribe: CUDA gate ──────────────────────────────────────────────────


def test_transcribe_raises_requires_cuda_when_no_gpu(monkeypatch):
    engine = ParakeetEngine()
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda_available=False))
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: False
    )
    with pytest.raises(ParakeetRequiresCuda):
        engine.transcribe("fake.wav")


# ── transcribe: near-silence short-circuit ─────────────────────────────────


def test_transcribe_short_circuits_on_near_silence(monkeypatch):
    engine = ParakeetEngine()
    model_loaded = []
    monkeypatch.setattr(engine, "_ensure_models", lambda: model_loaded.append(True))
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: True
    )

    result = engine.transcribe("silence.wav")

    # The model loader must never run for a near-silent recording.
    assert model_loaded == []
    assert len(result.segments) == 1
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 0.0
    assert result.segments[0].text == ""
    assert result.speaker_embeddings == {}


# ── transcribe: segment building + speaker assignment (faked NeMo) ─────────


def test_transcribe_builds_segments_and_assigns_speakers(monkeypatch):
    hypotheses = [
        FakeHypothesis(
            timestamp={
                "segment": [
                    {"segment": "Hello there", "start": 0.0, "end": 1.5},
                    {"segment": "And welcome", "start": 1.5, "end": 3.0},
                ]
            }
        )
    ]
    asr_model = FakeASRModel(hypotheses)
    turns = [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")]
    embeddings = {"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]}
    _install_stack(monkeypatch, asr_model=asr_model, diarize_turns=turns, embeddings=embeddings)
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: False
    )

    engine = ParakeetEngine()
    result = engine.transcribe("fake.wav")

    # The model was loaded from the canonical checkpoint name and transcribed
    # with timestamps=True.
    assert engine._asr_model is asr_model
    assert asr_model.transcribe_calls == [(["fake.wav"], True)]

    assert [s.text for s in result.segments] == ["Hello there", "And welcome"]
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [(s.start, s.end) for s in result.segments] == [(0.0, 1.5), (1.5, 3.0)]
    assert result.speaker_embeddings == embeddings


def test_transcribe_skips_empty_segment_entries(monkeypatch):
    hypotheses = [
        FakeHypothesis(
            timestamp={
                "segment": [
                    {"segment": "  ", "start": 0.0, "end": 0.5},  # whitespace -> skipped
                    {"segment": "Real words", "start": 0.5, "end": 1.0},
                ]
            }
        )
    ]
    asr_model = FakeASRModel(hypotheses)
    _install_stack(
        monkeypatch, asr_model=asr_model, diarize_turns=[(0.0, 1.0, "SPEAKER_00")], embeddings={}
    )
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: False
    )

    result = ParakeetEngine().transcribe("fake.wav")

    assert [s.text for s in result.segments] == ["Real words"]


def test_transcribe_warns_on_non_english_language(monkeypatch, caplog):
    hypotheses = [FakeHypothesis(timestamp={"segment": []})]
    asr_model = FakeASRModel(hypotheses)
    _install_stack(monkeypatch, asr_model=asr_model, diarize_turns=[], embeddings={})
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: False
    )

    with caplog.at_level("WARNING"):
        ParakeetEngine().transcribe("fake.wav", language="fr")

    assert any("English-only" in rec.message for rec in caplog.records)


def test_engine_stores_initial_prompt():
    """The constructor accepts ``initial_prompt`` for interface parity with
    WhisperXEngine, storing a stripped value (empty stays empty)."""
    assert ParakeetEngine()._initial_prompt == ""
    assert ParakeetEngine(initial_prompt="  field catalogue  ")._initial_prompt == "field catalogue"


def test_transcribe_warns_on_initial_prompt(monkeypatch, caplog):
    """A non-empty ``initial_prompt`` is a no-op but must warn (not fail) —
    Parakeet TDT has no prompt-conditioning mechanism, the same posture as
    ``custom_vocabulary``/``hotwords``."""
    hypotheses = [FakeHypothesis(timestamp={"segment": []})]
    asr_model = FakeASRModel(hypotheses)
    _install_stack(monkeypatch, asr_model=asr_model, diarize_turns=[], embeddings={})
    monkeypatch.setattr(
        "artifice_transcribe.services.parakeet_engine.is_near_silent", lambda p: False
    )

    with caplog.at_level("WARNING"):
        result = ParakeetEngine(initial_prompt="field catalogue").transcribe("fake.wav")

    # Completed without raising, building no segments from an empty timestamp set.
    assert result.segments == []
    assert any("initial_prompt" in rec.message for rec in caplog.records)


def test_default_model_name_is_the_registry_checkpoint():
    # Keep in step with model_harness.registry.ASR_MODELS["parakeet-tdt-1.1b"].
    assert DEFAULT_MODEL_NAME == "nvidia/parakeet-tdt-1.1b"


@pytest.mark.skipif(not _NEMO_AVAILABLE, reason="NeMo not installed — no real model load")
def test_real_nemo_raises_requires_cuda_without_gpu():
    """The one test that needs real NeMo: with NeMo installed but no GPU, the
    real ``_ensure_models`` CUDA gate must raise ParakeetRequiresCuda."""
    import torch

    if torch.cuda.is_available():
        pytest.skip("GPU present — CUDA gate not exercised")
    engine = ParakeetEngine()
    with pytest.raises(ParakeetRequiresCuda):
        engine._ensure_models()

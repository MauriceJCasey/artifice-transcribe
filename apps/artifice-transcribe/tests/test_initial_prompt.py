# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Whisper ``initial_prompt`` domain-vocabulary field (Item B).

Whisper's ``initial_prompt`` decoder-conditioning parameter — a plain-language
description of the collection, e.g. "a 19th-century archaeological field
catalogue in German Kurrentschrift" — is a *global Settings default* applied to
every job, distinct from the per-job ``hotwords``/``custom_vocabulary``
mechanism that already exists on the same faster-whisper ``TranscriptionOptions``
object.

These tests cover the surfaces the field touches:

* :class:`WhisperXEngine` sets ``options.initial_prompt`` (empty -> None, else
  the stripped value);
* the ``/config`` GET/PATCH round-trip and engine reload on change;
* per-job citation recording in the job's ``options`` sidecar.

(Parakeet's no-op/warning posture lives in ``test_parakeet_engine.py``, beside
the fake dependency stack it shares.)

``torch``/``whisperx`` are not installed in this dev environment or in CI, so
the :class:`WhisperXEngine` tests fake both into ``sys.modules`` — the same
technique ``test_parakeet_engine.py`` uses — rather than being skipped like the
conformance test in ``test_asr_backend.py`` (which guards on a real torch).
"""

from __future__ import annotations

import json
import sys
import types

import pytest
from artifice_transcribe.config import settings

# ── Fake dependency stack for WhisperXEngine ────────────────────────────────


def _fake_torch() -> types.ModuleType:
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    return torch


def _fake_whisperx() -> types.ModuleType:
    """A whisperx with only the pieces ``WhisperXEngine.transcribe`` reaches:
    ``align`` and ``assign_word_speakers``."""
    whisperx = types.ModuleType("whisperx")

    def _align(segments, align_model, metadata, audio_path, device):
        return {"segments": segments, "language": "en"}

    def _assign(diarize_segments, result):
        return result

    whisperx.align = _align
    whisperx.assign_word_speakers = _assign
    return whisperx


class _FakeOptions:
    """Stands in for faster-whisper's ``TranscriptionOptions`` (the object the
    engine mutates in place). ``hotwords`` and ``initial_prompt`` are separate
    fields, mirroring the real options object."""

    def __init__(self) -> None:
        self.hotwords = None
        self.initial_prompt = None


class _FakeWhisperModel:
    def __init__(self) -> None:
        self.options = _FakeOptions()

    def transcribe(self, audio_path, **kwargs):
        return {"segments": [], "language": "en"}


class _FakeDiarizeModel:
    def __call__(
        self, audio_path, *, min_speakers=None, max_speakers=None, return_embeddings=False
    ):
        return [], {}


def _build_whisperx_engine(monkeypatch, *, initial_prompt: str = ""):
    """Import :class:`WhisperXEngine` with torch/whisperx faked, and stub every
    heavy step so ``transcribe()`` runs to completion with no weights."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "whisperx", _fake_whisperx())

    from artifice_transcribe.services.transcription import WhisperXEngine

    monkeypatch.setattr(
        "artifice_transcribe.services.transcription.is_near_silent", lambda p: False
    )

    engine = WhisperXEngine(model_size="base", device="cpu", initial_prompt=initial_prompt)
    engine._whisper_model = _FakeWhisperModel()
    engine._diarize_model = _FakeDiarizeModel()
    monkeypatch.setattr(engine, "_ensure_models", lambda: None)
    monkeypatch.setattr(engine, "_get_align_model", lambda lang: (None, {}))
    return engine


# ── WhisperXEngine: options.initial_prompt ──────────────────────────────────


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("", None),  # empty behaves as "no prompt"
        ("field catalogue", "field catalogue"),
        ("  padded catalogue  ", "padded catalogue"),  # stripped at construction
    ],
)
def test_whisperx_engine_sets_options_initial_prompt(monkeypatch, prompt, expected):
    engine = _build_whisperx_engine(monkeypatch, initial_prompt=prompt)

    result = engine.transcribe("fake.wav")

    # Ran the full (faked) pipeline without the near-silence short-circuit.
    assert result.segments == []
    assert engine._whisper_model.options.initial_prompt == expected
    # The hotwords field is untouched by the initial_prompt path.
    assert engine._whisper_model.options.hotwords is None


# ── /config GET/PATCH round-trip ────────────────────────────────────────────


@pytest.mark.asyncio
class TestInitialPromptConfig:
    """GET/PATCH /api/v1/config must expose and honour ``whisper_initial_prompt``
    the same way it does ``whisper_model``/``asr_backend`` — including an engine
    reload on change, since the value is read at construction time."""

    async def test_get_config_includes_whisper_initial_prompt(self, api):
        resp = await api.client.get("/api/v1/config")
        assert resp.status_code == 200
        assert "whisper_initial_prompt" in resp.json()

    async def test_patch_whisper_initial_prompt_triggers_reload(self, api, monkeypatch):
        # Snapshot/restore the singleton so this test cannot leak a change into
        # later tests.
        monkeypatch.setattr(settings, "whisper_initial_prompt", settings.whisper_initial_prompt)

        reloaded = []

        async def _reload():
            reloaded.append(True)

        monkeypatch.setattr("artifice_transcribe.api.v1.routes._reload_engine", _reload)

        resp = await api.client.patch(
            "/api/v1/config", json={"whisper_initial_prompt": "field catalogue"}
        )
        assert resp.status_code == 200
        assert reloaded == [True], "changing whisper_initial_prompt must reload the engine"
        assert settings.whisper_initial_prompt == "field catalogue"

        # GET reflects the new value.
        resp = await api.client.get("/api/v1/config")
        assert resp.json()["whisper_initial_prompt"] == "field catalogue"


# ── Per-job citation recording ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestInitialPromptRecording:
    async def test_create_transcription_records_initial_prompt(self, api, monkeypatch):
        """POST /transcribe persists ``initial_prompt`` in the job's options
        sidecar, for methods-section citation (same posture as the ``mode``
        field in the same dict)."""
        from artifice_transcribe.db.models import TranscriptionJob
        from artifice_transcribe.services.asr_backend import TranscriptionResult

        monkeypatch.setattr(settings, "whisper_initial_prompt", "field catalogue")

        class FakeEngine:
            def transcribe(self, audio_path, **kwargs):
                return TranscriptionResult(segments=[], speaker_embeddings={})

            def unload(self) -> None: ...

        engine = FakeEngine()
        monkeypatch.setattr("artifice_transcribe.api.v1.routes._get_engine", lambda: engine)

        resp = await api.client.post(
            "/api/v1/transcribe",
            files={"file": ("interview.wav", b"fake-audio-data")},
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        async with api.session_factory() as session:
            job = await session.get(TranscriptionJob, job_id)
            options = json.loads(job.options)

        assert options["initial_prompt"] == "field catalogue"

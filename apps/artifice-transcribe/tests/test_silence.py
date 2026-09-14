# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Near-silence detection: Whisper hallucinates confident text over silence,
room tone and tape hiss, and oral-history recordings are full of long
silences. A cheap RMS check skips the ASR call entirely for a near-silent
recording — the audio analogue of OCR's near-blank page skip.

These tests never import the ASR stack (torch / whisperx / pyannote.audio),
so they run on a base install. The decode step (`_load_audio`) is stood in
for by a stdlib-`wave` decoder; the real one is Whisper's ffmpeg-backed
`load_audio`, which only exists once the ASR extra is installed.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest
from artifice_transcribe._silence import _rms, is_near_silent

_SAMPLE_RATE = 16000


def _write_wav(path, samples: np.ndarray) -> None:
    """Write a mono 16-bit PCM WAV from float32 samples in [-1, 1]."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def _read_wav_float32(path) -> np.ndarray:
    """Read a mono 16-bit PCM WAV back into float32 in [-1, 1].

    Test stand-in for ``whisper.audio.load_audio`` (same contract: 16 kHz mono
    float32), so the silence check is exercised end-to-end without the ASR
    stack installed.
    """
    with wave.open(str(path), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768.0


@pytest.fixture
def stdlib_decoder(monkeypatch):
    """Route ``_silence._load_audio`` through the stdlib WAV decoder above."""
    import artifice_transcribe._silence as silence

    monkeypatch.setattr(silence, "_load_audio", _read_wav_float32)
    return silence


# ── RMS math ───────────────────────────────────────────────────────────────


def test_rms_of_silence_is_zero():
    assert _rms(np.zeros(1000, dtype=np.float32)) == 0.0


def test_rms_of_constant_signal_is_its_amplitude():
    assert _rms(np.full(1000, 0.5, dtype=np.float32)) == pytest.approx(0.5)


def test_rms_of_empty_array_is_zero():
    assert _rms(np.array([], dtype=np.float32)) == 0.0


# ── is_near_silent ─────────────────────────────────────────────────────────


def test_digital_silence_is_near_silent(tmp_path, stdlib_decoder):
    path = tmp_path / "silence.wav"
    _write_wav(path, np.zeros(_SAMPLE_RATE, dtype=np.float32))
    assert is_near_silent(path) is True


def test_sine_tone_is_not_near_silent(tmp_path, stdlib_decoder):
    path = tmp_path / "tone.wav"
    t = np.arange(_SAMPLE_RATE, dtype=np.float32) / _SAMPLE_RATE
    tone = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    _write_wav(path, tone.astype(np.float32))
    assert is_near_silent(path) is False


def test_noise_is_not_near_silent(tmp_path, stdlib_decoder):
    path = tmp_path / "noise.wav"
    rng = np.random.default_rng(0)
    _write_wav(path, (0.3 * rng.standard_normal(_SAMPLE_RATE)).astype(np.float32))
    assert is_near_silent(path) is False


def test_undecodable_file_is_not_treated_as_silent(tmp_path, stdlib_decoder):
    """A decode failure must never silently skip a real recording."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a real wav file")
    assert is_near_silent(path) is False


def test_corrupt_file_never_raises_without_a_decoder(tmp_path):
    """The real ``_load_audio`` fails gracefully too: on a base install the
    lazy whisper import raises, and with whisper present a corrupt file makes
    ffmpeg raise — either way the check returns ``False``, never raises."""
    path = tmp_path / "corrupt.wav"
    path.write_bytes(b"not a real wav file")
    assert is_near_silent(path) is False


def test_custom_threshold_can_override_default(tmp_path, stdlib_decoder):
    path = tmp_path / "quiet.wav"
    _write_wav(path, np.full(_SAMPLE_RATE, 0.05, dtype=np.float32))  # RMS 0.05
    assert is_near_silent(path) is False  # above the 0.01 default
    assert is_near_silent(path, rms_threshold=0.1) is True  # below a raised bar

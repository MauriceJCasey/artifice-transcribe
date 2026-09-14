# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Near-silence detection.

Whisper is documented to hallucinate confident text over silence, room tone
and tape hiss — a genuine failure mode for oral-history recordings, which
routinely contain long silences, blank tape and machine hiss. The cheapest
fix is the same one OCR uses for blank pages: never send the model a
recording there is nothing to hear in. A near-silent recording has near-zero
RMS on its waveform regardless of how it was captured, and computing that
needs no model — it is safe to run unconditionally, before the ASR stack is
even loaded.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def _load_audio(audio_path: str) -> np.ndarray:
    """Decode *audio_path* to a 16 kHz mono float32 waveform in [-1, 1].

    Reuses WhisperX's own ffmpeg-backed loader — ``whisperx.audio.load_audio``,
    the same decode path its alignment and diarization steps already use —
    rather than inventing a second one. Imported lazily so this module stays
    importable on a base install where the ASR stack (whisperx/torch) is
    absent: the check only matters on the path that is about to call Whisper
    anyway.
    """
    from whisperx.audio import load_audio

    return load_audio(audio_path)


def _rms(audio: np.ndarray) -> float:
    """Root-mean-square of a mono float32 waveform in [-1, 1]."""
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def is_near_silent(audio_path: str | Path, *, rms_threshold: float = 0.01) -> bool:
    """True if *audio_path* decodes to a near-silent waveform.

    ``rms_threshold`` is a root-mean-square on a [-1, 1] waveform scale, where
    1.0 is a full-scale sine. Digital silence decodes to exactly 0.0; room
    tone and tape hiss on a normally-gained recording sit under ~0.005
    (~-46 dBFS); the quietest intelligible speech sits above ~0.02
    (~-34 dBFS). 0.01 (~-40 dBFS) leaves a wide margin on both sides. This
    default is provisional — there is no corpus yet (measurement item 0, the
    WER/CER harness) — and it is deliberately biased toward *not* skipping: a
    false positive silently discards a real recording, which is worse than a
    false negative that hallucinates a little on hiss. Never raises: an
    undecodable file returns ``False`` so a corrupt or unusual file is never
    mistaken for silent and silently skipped.
    """
    try:
        audio = _load_audio(str(audio_path))
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("Could not decode audio for silence check: %s", exc)
        return False
    return _rms(audio) <= rms_threshold

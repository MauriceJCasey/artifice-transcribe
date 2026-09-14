# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""NVIDIA Parakeet (NeMo) implementation of :class:`ASRBackend`.

Parakeet TDT 1.1B is a CUDA-only, English-only speech-to-text model shipped as
a single ``.nemo`` checkpoint on Hugging Face (``nvidia/parakeet-tdt-1.1b``,
CC-BY-4.0).  This module wraps it behind the same :class:`ASRBackend` contract
as :class:`artifice_transcribe.services.transcription.WhisperXEngine`, so the
rest of the app never learns which engine produced a transcript.

Design notes that differ from the WhisperX engine, deliberately:

* **No alignment step.**  ``transcribe(..., timestamps=True)`` already returns
  sentence-level timestamps in ``hypothesis.timestamp['segment']`` (each entry
  carries ``start``/``end`` in seconds and the text under the ``segment`` key),
  so there is no analogue of WhisperX's ``whisperx.align()`` call.  Segments are
  built directly from ``timestamp['segment']``.
* **CUDA-only.**  The maintainer chose not to build a CPU fallback.  If NeMo is
  installed but no GPU is present, model loading raises
  :class:`ParakeetRequiresCuda` rather than running a 4.3 GB model on CPU.
* **No diarization of its own.**  Speaker diarization stays pyannote-based
  regardless of ASR engine, reached through WhisperX's thin
  ``whisperx.diarize.DiarizationPipeline`` wrapper (reimplementing that would
  be pointless).  Segment→speaker assignment is a simple interval-overlap max,
  not word-level assignment — see :meth:`transcribe`.
* **English-only.**  There is no language parameter to set; a non-``en``
  request is logged and ignored rather than hard-failing (callers may pass a
  stale default).

The module is importable on a base install: ``nemo``, ``torch`` and ``whisperx``
are imported lazily inside :meth:`_ensure_models`, the same convention as the
rest of this app's ASR-optional imports.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np

from artifice_transcribe._silence import is_near_silent

from .asr_backend import Segment, TranscriptionResult
from .token_redaction import redact_token

logger = logging.getLogger(__name__)

# The only Parakeet checkpoint this app currently knows about (see
# model_harness.registry.ASR_MODELS["parakeet-tdt-1.1b"]).  There is no
# ``parakeet_model`` Settings field — Parakeet has a single canonical model.
DEFAULT_MODEL_NAME = "nvidia/parakeet-tdt-1.1b"

_FALLBACK_SPEAKER = "SPEAKER_00"


class ParakeetRequiresCuda(Exception):
    """Raised when Parakeet is selected but no CUDA GPU is available.

    Mirrors ``routes.AsrUnavailable``: the ``public_message`` attribute is set
    from a string literal, never from a wrapped third-party exception's
    ``str()`` (see ``AsrUnavailable``'s docstring — CodeQL's taint tracker
    flags the alternative).  Catch sites should read ``public_message`` rather
    than ``str(e)``.
    """

    def __init__(self) -> None:
        self.public_message = (
            "Parakeet requires a CUDA GPU, but none was detected. "
            "Switch to the WhisperX backend (`uv sync --extra asr`), which can "
            "run on CPU, or install the CUDA build with "
            "`uv sync --extra asr-cuda --extra asr-parakeet`."
        )
        super().__init__(self.public_message)


def _assign_speaker_by_overlap(
    seg_start: float,
    seg_end: float,
    turns: list[tuple[float, float, str]],
) -> str:
    """Return the speaker whose diarization turn overlaps a span the most.

    Algorithm (documented for the maintainer, not performance): for each
    diarization turn, compute the intersection duration with the segment —
    ``max(0, min(seg_end, turn_end) - max(seg_start, turn_start))`` — and keep
    the turn with the largest overlap.  A strictly-greater comparison means the
    earliest turn wins ties.  If no turn overlaps the segment (e.g. diarization
    produced nothing), fall back to ``SPEAKER_00``.

    This is deliberately coarse: Parakeet already yields sentence-level
    segments, so a per-segment overlap max is all that is needed — there is no
    per-word assignment the way WhisperX does.
    """
    best_speaker = _FALLBACK_SPEAKER
    best_overlap = 0.0
    for turn_start, turn_end, speaker in turns:
        overlap = min(seg_end, turn_end) - max(seg_start, turn_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = speaker
    return best_speaker


class ParakeetEngine:
    """Parakeet (NeMo) implementation of :class:`ASRBackend`.

    Construction only records settings — no model is loaded, and no CUDA probe
    runs, until :meth:`_ensure_models` is first reached (the same lazy-loading
    contract as :class:`WhisperXEngine`).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        hf_token: str = "",
        diarization_model: str = "",
        initial_prompt: str = "",
    ):
        self._model_name = model_name
        self._device = self._resolve_device(device)
        # Used only for pyannote diarization, never for the ASR model itself.
        self._hf_token = hf_token
        self._diarization_model = (diarization_model or "").strip()
        # Accepted for interface parity with WhisperXEngine, but a no-op:
        # Parakeet TDT has no prompt-conditioning mechanism (same posture as
        # custom_vocabulary/hotwords). A non-empty value is warned about in
        # transcribe() rather than silently dropped.
        self._initial_prompt = (initial_prompt or "").strip()

        self._asr_model = None
        self._diarize_model = None
        self._models_ready = False
        self._last_error: str | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Parakeet is CUDA-only, so "auto" resolves to "cuda".

        An explicit device is still honored for the diarization side (pyannote
        can run on CPU), but the ASR model itself always runs on CUDA — the
        availability of which is enforced by :meth:`_ensure_models`.
        """
        if device == "auto":
            return "cuda"
        return device

    def _ensure_models(self) -> None:
        """Load the Parakeet ASR model and the diarization pipeline.

        All third-party imports happen here, lazily, so a base install can
        import this module with neither NeMo, torch nor whisperx present.  The
        CUDA gate runs before anything else: NeMo can be installed on a machine
        with no GPU, and that must surface as :class:`ParakeetRequiresCuda`,
        not as a slow (or broken) CPU load.
        """
        import torch

        if not torch.cuda.is_available():
            raise ParakeetRequiresCuda()

        if self._asr_model is None:
            try:
                import nemo.collections.asr as nemo_asr

                logger.info("Loading Parakeet model %s on cuda", self._model_name)
                self._asr_model = nemo_asr.models.ASRModel.from_pretrained(self._model_name)
            except Exception as exc:
                self._last_error = redact_token(str(exc))
                self._models_ready = False
                logger.error("Parakeet model loading failed: %s", redact_token(str(exc)))
                raise

        if self._diarize_model is None:
            try:
                from whisperx.diarize import DiarizationPipeline

                # Same "None, never ''" discipline as WhisperXEngine (see its
                # _ensure_models): an empty string is a present-but-blank
                # credential to huggingface_hub, which is not "no credential".
                model_name = self._diarization_model or None
                logger.info(
                    "Loading diarization model: %s",
                    model_name or "(WhisperX default)",
                )
                self._diarize_model = DiarizationPipeline(
                    model_name=model_name,
                    token=self._hf_token or None,
                    device=self._device,
                )
            except Exception as exc:
                self._last_error = redact_token(str(exc))
                self._models_ready = False
                logger.error("Diarization model loading failed: %s", redact_token(str(exc)))
                raise

        self._models_ready = True
        self._last_error = None

    def unload(self) -> None:
        self._asr_model = None
        self._diarize_model = None
        gc.collect()
        if self._device == "cuda":
            import torch

            torch.cuda.empty_cache()
        logger.info("Parakeet models unloaded and VRAM freed")

    def health_check(self) -> dict:
        """Return engine status with 3-state model info: untested / loaded / failed."""
        gpu_info = None
        try:
            import torch

            if torch.cuda.is_available():
                gpu_info = {
                    "name": torch.cuda.get_device_name(0),
                    "vram_total_mb": round(
                        torch.cuda.get_device_properties(0).total_memory / 1048576
                    ),
                    "vram_used_mb": round(torch.cuda.memory_allocated(0) / 1048576),
                    "vram_free_mb": round(torch.cuda.memory_reserved(0) / 1048576),
                }
        except ImportError:
            # torch absent (base install) — report unavailable rather than raise.
            pass

        if self._asr_model is not None:
            model_state = "loaded"
        elif self._last_error:
            model_state = "failed"
        else:
            model_state = "untested"

        if self._diarize_model is not None:
            diarize_state = "loaded"
        elif self._last_error:
            diarize_state = "failed"
        else:
            diarize_state = "untested"

        return {
            "parakeet_model": {
                "name": self._model_name,
                "state": model_state,
                "loaded": self._asr_model is not None,
            },
            "diarization_model": {
                "state": diarize_state,
                "loaded": self._diarize_model is not None,
            },
            "device": self._device,
            "gpu": gpu_info,
            "hf_token_configured": bool(self._hf_token),
            "last_error": self._last_error,
        }

    def preload(self) -> dict:
        """Attempt to load all models. Returns success or error."""
        try:
            self._ensure_models()
            return {"ok": True, "message": "All models loaded successfully"}
        except ParakeetRequiresCuda as exc:
            return {"ok": False, "error": exc.public_message}
        except Exception as exc:
            return {"ok": False, "error": redact_token(str(exc))}

    def extract_speaker_embedding(self, audio_path: str | Path) -> np.ndarray:
        """Return the raw pyannote speaker embedding for the given clip.

        Identical diarization machinery to ``WhisperXEngine`` — the embedding
        model is pyannote's regardless of which ASR engine produced the words.
        """
        self._ensure_models()
        embedder = self._diarize_model.model._embedding
        from pyannote.audio import Inference  # type: ignore[import-untyped]

        inference = Inference(embedder, window="whole")
        return inference(str(audio_path))

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
    ) -> TranscriptionResult:
        """Transcribe with timestamps, then diarize and assign speakers.

        Pipeline: (1) near-silence short-circuit, (2) load models, (3)
        ``transcribe(..., timestamps=True)``, (4) build ``Segment``s straight
        from ``timestamp['segment']`` (no alignment step — Parakeet supplies
        sentence timestamps directly), (5) diarize, (6) assign each segment the
        speaker with the greatest temporal overlap (see
        :func:`_assign_speaker_by_overlap`), (7) normalise per-speaker
        embeddings.

        ``custom_vocabulary`` and ``hotwords`` are accepted for interface
        parity but are no-ops: Parakeet TDT has no keyword-biasing mechanism.
        Same for ``initial_prompt`` (the constructor), which is logged and
        ignored when non-empty.
        """
        # Near-silence short-circuit — same convention as WhisperXEngine, and
        # easy to forget because it is not part of the ASRBackend protocol.
        if is_near_silent(audio_path):
            logger.warning("Skipping transcription: %s detected as near-silent", audio_path)
            return TranscriptionResult(
                segments=[Segment(speaker=_FALLBACK_SPEAKER, start=0.0, end=0.0, text="")],
                speaker_embeddings={},
            )

        # Parakeet is English-only: warn (do not fail) on a non-English request.
        if language is not None and language.lower() != "en":
            logger.warning(
                "Parakeet is English-only; ignoring requested language=%r and "
                "transcribing in English",
                language,
            )

        # Parakeet TDT has no prompt-conditioning mechanism: a non-empty
        # initial_prompt (the Whisper-only domain-vocabulary field) is logged
        # and ignored rather than hard-failing, the same posture as
        # custom_vocabulary/hotwords.
        if self._initial_prompt:
            logger.warning(
                "Parakeet TDT has no prompt-conditioning mechanism; ignoring initial_prompt=%r",
                self._initial_prompt,
            )

        self._ensure_models()
        audio_path = str(audio_path)

        if progress_callback:
            progress_callback(0.1)

        # 3. Transcribe with timestamps.
        logger.info("Transcribing %s with Parakeet", audio_path)
        hypotheses = self._asr_model.transcribe([audio_path], timestamps=True)
        if progress_callback:
            progress_callback(0.5)

        # 4. Build segments from timestamp['segment'] — no alignment step.
        #    Each entry is {"segment": text, "start": float, "end": float, ...}
        #    (verified against nemo.collections.asr.parts.utils.timestamp_utils).
        segment_entries = (hypotheses[0].timestamp or {}).get("segment", []) if hypotheses else []
        segments: list[Segment] = []
        for entry in segment_entries:
            text = str(entry.get("segment", "")).strip()
            if not text:
                continue
            segments.append(
                Segment(
                    speaker=_FALLBACK_SPEAKER,
                    start=float(entry.get("start", 0.0)),
                    end=float(entry.get("end", 0.0)),
                    text=text,
                )
            )

        # 5. Diarize (with embeddings for cross-session recognition).
        logger.info("Running diarization...")
        diarize_result = self._diarize_model(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=True,
        )
        diarize_segments, speaker_embeddings_raw = diarize_result
        if progress_callback:
            progress_callback(0.9)

        # 6. Assign each segment the speaker with the greatest temporal overlap.
        turns = [
            (float(turn.start), float(turn.end), str(turn.speaker))
            for turn in diarize_segments.itertuples(index=False)
        ]
        for seg in segments:
            seg.speaker = _assign_speaker_by_overlap(seg.start, seg.end, turns)

        segments.sort(key=lambda s: s.start)

        # 7. Normalise speaker embeddings into a label-keyed dict.
        embeddings: dict[str, list[float]] = {}
        if isinstance(speaker_embeddings_raw, dict):
            for label, emb in speaker_embeddings_raw.items():
                if isinstance(emb, (list, tuple)):
                    embeddings[label] = list(emb)

        if progress_callback:
            progress_callback(1.0)

        return TranscriptionResult(segments=segments, speaker_embeddings=embeddings)

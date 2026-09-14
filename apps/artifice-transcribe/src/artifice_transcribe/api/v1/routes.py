# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException
from model_harness.contract import EndpointRejected
from model_harness.endpoint_policy import EndpointPolicy
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from artifice_transcribe.config import settings
from artifice_transcribe.db.models import (
    JobStatus,
    KnownSpeaker,
    LegacyEmbeddingError,
    PersistentDictionary,
    SpeakerEmbedding,
    SpeakerMapping,
    TranscriptionJob,
    pack_embedding,
    unpack_embedding,
)
from artifice_transcribe.db.session import async_session
from artifice_transcribe.schemas.transcription import TranscriptionOptions
from artifice_transcribe.services.asr_backend import ASRBackend
from artifice_transcribe.services.inference import InferenceEngine  # noqa: F401
from artifice_transcribe.services.token_redaction import redact_token

# ``InferenceEngine`` is not used in this module directly (its only callers,
# ``summarize_job``/``cleanup_job``, moved to ``api/v1/transcription.py`` in
# Phase 6). It stays imported here, unused, because
# ``test_inference_client_close.py`` monkeypatches it via the string path
# ``artifice_transcribe.api.v1.routes.InferenceEngine.generate`` —
# ``monkeypatch.setattr`` resolves that path by attribute lookup, so
# ``routes.InferenceEngine`` must exist as a name on this module or the
# patch's own attribute resolution fails before it ever reaches the class.
# Removing this import does not change *what* gets patched (the class object
# is shared with transcription.py's own import of the same name), only
# whether the test can find it.

logger = logging.getLogger(__name__)


# ── Model endpoints ──────────────────────────────────────────────────────────
#
# The allowlist policy lives in ``model_harness.endpoint_policy`` — this app
# only wraps it with FastAPI's exception type.  This is the rule the harness
# exists to own; the duplicate that lived here before Phase 3 has been
# collapsed into the harness.
#
# See :class:`model_harness.endpoint_policy.EndpointPolicy` for the full
# rationale and constraint set.

_endpoint_policy = EndpointPolicy()


def _classify_host(host: str) -> tuple[bool, str]:
    """Return ``(permitted, reason)`` for a URL host.
    Delegates to the harness policy."""
    return _endpoint_policy.classify_host(host)


def _validate_base_url(raw: str, field_name: str) -> str:
    """Return *raw* after checking its scheme and host. Fails closed, loudly."""
    try:
        return _endpoint_policy.validate_url(raw)
    except EndpointRejected as e:
        raise HTTPException(status_code=400, detail=f"{field_name}: {e}") from e


# ── Inference configuration persistence helper ───────────────────────────────
# Uses platformdirs to resolve a per-user data directory, so the config file
# survives frozen bundles (.exe/.dmg) where CWD can be anywhere.
_LEGACY_INFERENCE_CONFIG = Path("./data/inference_config.json").resolve()
_LEGACY_PT_INFERENCE_CONFIG = Path("./data/pt-inference-config.json").resolve()
_INFERENCE_CONFIG_FILE = settings.data_path / "inference_config.json"
_HF_TOKEN_FILE = settings.data_path / "hf_token.json"


def _load_hf_token() -> str:
    """Return the HF token, preferring the env var and falling back to
    the secure-io-protected file on disk.

    The token is never read from the plaintext ``.env`` file through
    ``Settings.hf_token`` alone — if the env var is unset the secure-io
    file is consulted.  This keeps the Zero Secrets Policy self-consistent.
    """
    token = settings.hf_token
    if token:
        return token
    if _HF_TOKEN_FILE.exists():
        from secure_io import ensure_restricted

        ensure_restricted(_HF_TOKEN_FILE)
        try:
            data = json.loads(_HF_TOKEN_FILE.read_text(encoding="utf-8"))
            return data.get("hf_token", "")
        except Exception as exc:
            logger.warning("Could not read %s — using defaults: %s", _HF_TOKEN_FILE, exc)
    return ""


def _save_hf_token(token: str) -> None:
    """Persist the HF token through ``secure_io.write_private_json``."""
    from secure_io import write_private_json_verified

    _HF_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_verified(_HF_TOKEN_FILE, {"hf_token": token}, label="HF token file")


def _migrate_legacy_inference_config() -> None:
    """Move legacy ``./data/inference_config.json`` to the platform data dir."""
    if _INFERENCE_CONFIG_FILE.exists():
        if _LEGACY_INFERENCE_CONFIG.exists():
            logger.info(
                "Legacy inference config found at %s but config already exists at %s. "
                "Using the existing config.",
                _LEGACY_INFERENCE_CONFIG,
                _INFERENCE_CONFIG_FILE,
            )
        return
    if _LEGACY_INFERENCE_CONFIG.exists():
        logger.info(
            "Migrating legacy inference config from %s to %s",
            _LEGACY_INFERENCE_CONFIG,
            _INFERENCE_CONFIG_FILE,
        )
        _INFERENCE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.move(str(_LEGACY_INFERENCE_CONFIG), str(_INFERENCE_CONFIG_FILE))
        logger.info("Migration complete — inference config is now at %s", _INFERENCE_CONFIG_FILE)


def _load_inference_config() -> dict:
    _migrate_legacy_inference_config()
    if _INFERENCE_CONFIG_FILE.exists():
        from secure_io import ensure_restricted

        ensure_restricted(_INFERENCE_CONFIG_FILE)
        try:
            return json.loads(_INFERENCE_CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read %s — using defaults: %s", _INFERENCE_CONFIG_FILE, exc)
    return {
        "base_url": "http://localhost:11434/v1",
        "api_key": "not-needed",
        "model_name": "",
        "vision_enabled": False,
    }


def _save_inference_config(cfg: dict) -> None:
    from secure_io import write_private_json_verified

    _INFERENCE_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_private_json_verified(_INFERENCE_CONFIG_FILE, cfg, label="inference config")


# Module-level engine singleton (lazy init)
_engine: ASRBackend | None = None

_INSTALL_HINT = "uv sync --extra asr"
_INSTALL_HINT_CUDA = "uv sync --extra asr-cuda"

# ASR backends selectable via settings.asr_backend. Kept in step with
# ModelConfigResponse.available_asr_backends (schemas/transcription.py).
_ASR_BACKENDS = frozenset({"whisperx", "parakeet"})


class AsrUnavailable(Exception):
    """Raised when the ASR stack (torch, whisperx, pyannote) is not installed.

    The ``public_message`` attribute is set from string literals — it is
    never derived from a wrapped third-party exception.  Catch sites should
    read ``public_message`` for the response body rather than calling
    ``str(e)``, which CodeQL's taint tracker treats as unsafe.
    """

    def __init__(self, extra: str = "asr") -> None:
        hint = _INSTALL_HINT_CUDA if extra == "asr-cuda" else _INSTALL_HINT
        self.public_message = (
            f"The transcription stack is not installed. "
            f"Run `{hint}` to install it, then restart the server."
        )
        super().__init__(self.public_message)


def _build_engine() -> ASRBackend:
    """Construct the ASR backend selected by ``settings.asr_backend``.

    This is the seam the :class:`ASRBackend` protocol was built for: both
    engines share ``device`` / ``hf_token`` / ``diarization_model``, and differ
    only in their model identifier.  The heavy imports happen here (and only
    here) so a base install can import this module; an absent stack surfaces as
    :class:`AsrUnavailable`, as it always has.
    """
    try:
        if settings.asr_backend == "parakeet":
            from artifice_transcribe.services.parakeet_engine import ParakeetEngine

            return ParakeetEngine(
                device=settings.device,
                hf_token=_load_hf_token(),
                diarization_model=settings.diarization_model,
                initial_prompt=settings.whisper_initial_prompt,
            )
        from artifice_transcribe.services.transcription import WhisperXEngine

        return WhisperXEngine(
            model_size=settings.whisper_model,
            device=settings.device,
            hf_token=_load_hf_token(),
            diarization_model=settings.diarization_model,
            initial_prompt=settings.whisper_initial_prompt,
        )
    except ImportError as exc:
        raise AsrUnavailable() from exc


async def _reload_engine():
    """Rebuild the engine in place, unloading the previous one first.

    Generalized from the old ``_reload_engine_with_new_model``, which took a
    WhisperX model-size string as an argument.  With two backends the engine
    identity is no longer a single model string, so the reload now reads
    ``settings.asr_backend`` (and ``settings.whisper_model``) directly and the
    call site simply checks *which* fields changed.
    """
    global _engine
    logger.info(
        "Reloading engine (backend=%s, whisper_model=%s)",
        settings.asr_backend,
        settings.whisper_model,
    )

    old_engine = _engine
    if old_engine:
        old_engine.unload()

    _engine = _build_engine()
    logger.info("Engine reloaded successfully")


def _get_engine():
    global _engine
    if _engine is None:
        _engine = _build_engine()
    return _engine


# ── Background worker ────────────────────────────────────────────────────────


async def _run_transcription(
    job_id: str,
    audio_path: str,
    options: TranscriptionOptions,
) -> None:
    """Background task: run transcription engine, update DB on completion/failure."""
    async with async_session() as db:
        try:
            job = await db.get(TranscriptionJob, job_id)
            if job is None:
                return
            job.status = JobStatus.processing
            job.progress_percentage = 5.0
            await db.commit()

            def _progress(pct: float) -> None:
                logger.debug("Job %s progress: %.0f%%", job_id, pct * 100)

            engine = _get_engine()

            # Merge global persistent dictionary with per-job vocabulary
            hotwords = None
            dict_row = (await db.execute(select(PersistentDictionary).limit(1))).scalars().first()
            if dict_row and dict_row.words:
                hotwords = dict_row.words
            custom_vocab = job.custom_vocabulary or None

            result = await asyncio.to_thread(
                engine.transcribe,
                audio_path,
                language=options.language,
                min_speakers=options.min_speakers,
                max_speakers=options.max_speakers,
                progress_callback=_progress,
                custom_vocabulary=custom_vocab,
                hotwords=hotwords,
            )
            segments = result.segments
            speaker_embeddings = result.speaker_embeddings

            from artifice_transcribe.db.models import TranscriptSegment as TS

            db_segs = [
                TS(
                    job_id=job_id,
                    speaker_label=seg.speaker,
                    start_time=seg.start,
                    end_time=seg.end,
                    text=seg.text,
                )
                for seg in segments
            ]
            db.add_all(db_segs)

            seen = []
            for seg in segments:
                if seg.speaker not in seen:
                    seen.append(seg.speaker)

            existing = (
                (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
                .scalars()
                .all()
            )
            existing_labels = {m.speaker_label for m in existing}

            new_mappings = [
                SpeakerMapping(job_id=job_id, speaker_label=label, custom_name=label)
                for label in seen
                if label not in existing_labels
            ]
            db.add_all(new_mappings)

            # Store speaker embeddings for cross-session matching
            if speaker_embeddings:
                db_embeddings = [
                    SpeakerEmbedding(
                        job_id=job_id,
                        speaker_label=label,
                        embedding=pack_embedding(emb),
                        model_name="pyannote/embedding",
                        dimension=len(emb),
                    )
                    for label, emb in speaker_embeddings.items()
                ]
                db.add_all(db_embeddings)

            job.status = JobStatus.completed
            job.progress_percentage = 100.0
            job.completed_at = datetime.now(UTC)
            await db.commit()
            logger.info("Job %s completed with %d segments", job_id, len(segments))

            # Auto-match speakers against known speakers
            try:
                await _auto_match_speakers(job_id, db)
            except Exception:
                logger.exception("Auto-match failed for job %s", job_id)

        except Exception as exc:
            logger.error("Job %s failed: %s", job_id, redact_token(str(exc)))
            job = await db.get(TranscriptionJob, job_id)
            if job:
                job.status = JobStatus.failed
                job.error_message = redact_token(str(exc))
                job.completed_at = datetime.now(UTC)
                await db.commit()

        finally:
            try:
                engine = _get_engine()
                if engine is not None:
                    engine.unload()
            except AsrUnavailable:
                pass


async def _auto_match_speakers(job_id: str, db: AsyncSession) -> None:
    """Compare this job's speaker embeddings against known speakers and
    auto-rename mappings when a match exceeds the confidence threshold."""
    embeddings = (
        (await db.execute(select(SpeakerEmbedding).where(SpeakerEmbedding.job_id == job_id)))
        .scalars()
        .all()
    )
    if not embeddings:
        return

    known = (await db.execute(select(KnownSpeaker))).scalars().all()
    if not known:
        return

    import numpy as np

    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    THRESHOLD = 0.65

    for emb in embeddings:
        emb_vec = unpack_embedding(emb.embedding, emb.dimension)

        best_name = None
        best_score = -1.0

        for known_spk in known:
            try:
                known_vec = unpack_embedding(known_spk.embedding, known_spk.dimension)
            except LegacyEmbeddingError:
                logger.warning(
                    "Skipping known speaker '%s' (id=%s): embedding predates "
                    "format change, must be re-enrolled",
                    known_spk.name,
                    known_spk.id,
                )
                continue
            score = _cosine_sim(emb_vec, known_vec)
            if score > best_score:
                best_score = score
                best_name = known_spk.name

        if best_score >= THRESHOLD and best_name:
            mapping = (
                (
                    await db.execute(
                        select(SpeakerMapping).where(
                            SpeakerMapping.job_id == job_id,
                            SpeakerMapping.speaker_label == emb.speaker_label,
                        )
                    )
                )
                .scalars()
                .first()
            )
            if mapping:
                mapping.custom_name = best_name

    await db.commit()

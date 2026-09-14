# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from shared_ui.path_validation import (
    PathValidationError,
    assert_contained,
    sanitise_path_component,
)
from shared_ui.uploads import UploadTooLarge, read_capped_to_tempfile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from artifice_transcribe.api.v1 import routes as _routes
from artifice_transcribe.api.v1.routes import _validate_base_url
from artifice_transcribe.config import settings
from artifice_transcribe.db.models import JobStatus, TranscriptionJob, TranscriptSegment
from artifice_transcribe.db.session import get_db
from artifice_transcribe.schemas.transcription import JobCreated, TranscriptionOptions
from artifice_transcribe.services.inference import InferenceEngine
from artifice_transcribe.services.token_redaction import redact_token

router = APIRouter(prefix="/api/v1", tags=["transcription"])


async def _build_transcript_prompt(job_id: str, db: AsyncSession, action: str) -> str:
    """Fetch segments for a job and build a prompt for the AI action."""
    segs = (
        (
            await db.execute(
                select(TranscriptSegment)
                .where(TranscriptSegment.job_id == job_id)
                .order_by(TranscriptSegment.start_time)
            )
        )
        .scalars()
        .all()
    )
    if not segs:
        return ""

    lines = []
    for s in segs:
        start = int(s.start_time // 60)
        end_sec = int(s.end_time % 60)
        start_sec = int(s.start_time % 60)
        lines.append(f"[{start}:{start_sec:02d}-{start}:{end_sec:02d}] {s.speaker_label}: {s.text}")

    transcript = "\n".join(lines)

    if action == "summarize":
        return (
            "Provide a clear, structured summary of the following transcript. "
            "Include: (1) a brief overview paragraph, "
            "(2) key topics discussed, "
            "(3) any notable quotes or decisions, "
            "and (4) a list of action items if any are mentioned. "
            "Preserve the meaning and speaker context.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
    elif action == "cleanup":
        return (
            "Clean up the following transcript by removing verbal tics, "
            "disfluencies (um, ah, uh, like, you know), false starts, "
            "and stuttering. Preserve the exact speaker attribution, "
            "core meaning, key names, and proper nouns. "
            "Fix punctuation and capitalization where needed. "
            "Output the cleaned transcript in the same format.\n\n"
            f"TRANSCRIPT:\n{transcript}"
        )
    return transcript


@router.post("/jobs/{job_id}/summarize")
async def summarize_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Stream a summary of the transcript for the given job."""
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

    prompt = await _build_transcript_prompt(job_id, db, "summarize")
    if not prompt:
        raise HTTPException(400, "No transcript segments found for this job")

    cfg = _routes._load_inference_config()
    _validate_base_url(cfg.get("base_url", "http://localhost:11434/v1"), "base_url")
    engine = InferenceEngine(
        base_url=cfg.get("base_url", "http://localhost:11434/v1"),
        api_key=cfg.get("api_key", "not-needed"),
        model_name=cfg.get("model_name", ""),
        vision_enabled=cfg.get("vision_enabled", False),
    )

    async def stream_generator():
        try:
            gen = await engine.generate(
                prompt=prompt,
                stream=True,
                temperature=0.3,
                max_tokens=2048,
            )
            async for chunk in gen:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': redact_token(str(exc))})}\n\n"
        finally:
            await engine.aclose()
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/jobs/{job_id}/cleanup")
async def cleanup_job(job_id: str, db: AsyncSession = Depends(get_db)):
    """Stream a cleaned-up version of the transcript for the given job."""
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

    prompt = await _build_transcript_prompt(job_id, db, "cleanup")
    if not prompt:
        raise HTTPException(400, "No transcript segments found for this job")

    cfg = _routes._load_inference_config()
    _validate_base_url(cfg.get("base_url", "http://localhost:11434/v1"), "base_url")
    engine = InferenceEngine(
        base_url=cfg.get("base_url", "http://localhost:11434/v1"),
        api_key=cfg.get("api_key", "not-needed"),
        model_name=cfg.get("model_name", ""),
        vision_enabled=cfg.get("vision_enabled", False),
    )

    async def stream_generator():
        try:
            gen = await engine.generate(
                prompt=prompt,
                stream=True,
                temperature=0.2,
                max_tokens=4096,
            )
            async for chunk in gen:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'text': redact_token(str(exc))})}\n\n"
        finally:
            await engine.aclose()
        yield 'data: {"type": "done"}\n\n'

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.post("/transcribe", response_model=JobCreated, status_code=202)
async def create_transcription(
    file: UploadFile,
    language: str | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    custom_vocabulary: str | None = None,
    mode: str = "auto",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
) -> JobCreated:
    if mode not in ("auto", "manual"):
        raise HTTPException(status_code=400, detail="mode must be 'auto' or 'manual'")

    try:
        spooled = await read_capped_to_tempfile(file, settings.max_upload_size)
    except UploadTooLarge as e:
        raise HTTPException(status_code=413, detail=e.public_message) from e

    job = TranscriptionJob(
        filename=file.filename or "unknown",
        status=JobStatus.queued,
        custom_vocabulary=custom_vocabulary,
        options=json.dumps(
            {
                "mode": mode,
                "language": language,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "initial_prompt": settings.whisper_initial_prompt,
            }
        ),
    )
    db.add(job)
    await db.commit()

    try:
        safe_filename = sanitise_path_component(file.filename or "unknown")
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=e.public_message) from e
    audio_path = settings.upload_path / f"{job.id}_{safe_filename}"
    try:
        assert_contained(audio_path, settings.upload_path)
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=e.public_message) from e

    def _persist(spooled_file, dest_path):
        import shutil

        with spooled_file, open(dest_path, "wb") as out:
            shutil.copyfileobj(spooled_file, out)

    await asyncio.to_thread(_persist, spooled, audio_path)

    if mode == "manual":
        # Hand-transcription job: skip ASR entirely. The uploaded audio is
        # kept for the editor's audio player, but nothing is queued — the job
        # is already complete and seeded with one empty segment to type into.
        job.status = JobStatus.completed
        job.progress_percentage = 100.0
        job.completed_at = datetime.now(UTC)
        db.add(
            TranscriptSegment(
                job_id=job.id,
                speaker_label="SPEAKER_00",
                start_time=0.0,
                end_time=0.0,
                text="",
            )
        )
        await db.commit()
        return JobCreated(job_id=job.id, status=JobStatus.completed)

    opts = TranscriptionOptions(
        language=language,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    background_tasks.add_task(_routes._run_transcription, job.id, str(audio_path), opts)

    return JobCreated(job_id=job.id)


@router.post("/transcribe/batch", status_code=202)
async def create_batch_transcription(
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Upload multiple audio files for batch transcription."""

    upload_dir = settings.upload_path
    audio_exts = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac", "*.mp4", "*.m4v")
    files = []
    for ext in audio_exts:
        files.extend(upload_dir.glob(ext))

    queued = 0
    for fp in files:
        job = TranscriptionJob(
            filename=fp.name,
            status=JobStatus.queued,
            options=json.dumps({"language": None, "min_speakers": None, "max_speakers": None}),
        )
        db.add(job)
        await db.commit()
        opts = TranscriptionOptions(language=None, min_speakers=None, max_speakers=None)
        background_tasks.add_task(_routes._run_transcription, job.id, str(fp), opts)
        queued += 1

    return {"queued": queued, "message": f"Queued {queued} file(s) for transcription"}

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from shared_ui.path_validation import (
    PathValidationError,
    assert_contained,
    sanitise_path_component,
)
from shared_ui.uploads import UploadTooLarge, read_capped_to_tempfile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from artifice_transcribe.api.v1 import routes as _routes
from artifice_transcribe.api.v1.routes import AsrUnavailable, _auto_match_speakers
from artifice_transcribe.config import settings
from artifice_transcribe.db.models import (
    JobStatus,
    KnownSpeaker,
    SpeakerEmbedding,
    SpeakerMapping,
    TranscriptionJob,
    _is_legacy_pickle_blob,
)
from artifice_transcribe.db.session import get_db
from artifice_transcribe.schemas.transcription import (
    EnrollFromJobRequest,
    KnownSpeakerList,
    KnownSpeakerOut,
    SpeakerEmbeddingOut,
    SpeakerEnrollResponse,
    SpeakerMatchResponse,
    SpeakerMatchResult,
)
from artifice_transcribe.services.parakeet_engine import ParakeetRequiresCuda

router = APIRouter(prefix="/api/v1", tags=["speakers"])


@router.post("/speakers/enroll", response_model=SpeakerEnrollResponse)
async def enroll_speaker(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> SpeakerEnrollResponse:
    """Enroll a known speaker by uploading a short audio clip of their voice."""

    try:
        spooled = await read_capped_to_tempfile(file, settings.max_upload_size)
    except UploadTooLarge as e:
        raise HTTPException(status_code=413, detail=e.public_message) from e

    try:
        safe_name = sanitise_path_component(name, field_name="name")
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=e.public_message) from e
    try:
        safe_filename = sanitise_path_component(file.filename or "unknown")
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=e.public_message) from e
    audio_path = settings.upload_path / f"enroll_{safe_name}_{safe_filename}"
    try:
        assert_contained(audio_path, settings.upload_path)
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=e.public_message) from e

    def _persist(spooled_file, dest_path):
        import shutil

        with spooled_file, open(dest_path, "wb") as out:
            shutil.copyfileobj(spooled_file, out)

    await asyncio.to_thread(_persist, spooled, audio_path)

    try:
        engine = _routes._get_engine()
    except AsrUnavailable as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from exc

    try:
        embedding = engine.extract_speaker_embedding(audio_path)
    except ParakeetRequiresCuda as exc:
        raise HTTPException(status_code=503, detail=exc.public_message) from exc

    emb_bytes = _routes.pack_embedding(embedding)
    spk = KnownSpeaker(
        name=name,
        embedding=emb_bytes,
        model_name="pyannote/embedding",
        dimension=len(embedding),
        sample_audio_path=str(audio_path),
    )
    db.add(spk)
    await db.commit()

    return SpeakerEnrollResponse(id=spk.id, name=spk.name)


@router.post("/speakers/enroll-from-job", response_model=SpeakerEnrollResponse)
async def enroll_speaker_from_job(
    body: EnrollFromJobRequest,
    db: AsyncSession = Depends(get_db),
) -> SpeakerEnrollResponse:
    """Enroll a known speaker from a completed job's existing embedding."""

    emb = (
        (
            await db.execute(
                select(SpeakerEmbedding).where(
                    SpeakerEmbedding.job_id == body.job_id,
                    SpeakerEmbedding.speaker_label == body.speaker_label,
                )
            )
        )
        .scalars()
        .first()
    )
    if emb is None:
        raise HTTPException(
            404, f"No embedding found for {body.speaker_label} in job {body.job_id}"
        )

    spk = KnownSpeaker(
        name=body.name,
        embedding=emb.embedding,
        model_name=emb.model_name,
        dimension=emb.dimension,
    )
    db.add(spk)
    await db.commit()

    return SpeakerEnrollResponse(id=spk.id, name=spk.name)


@router.get("/speakers/known", response_model=KnownSpeakerList)
async def list_known_speakers(db: AsyncSession = Depends(get_db)) -> KnownSpeakerList:
    speakers = (await db.execute(select(KnownSpeaker))).scalars().all()
    return KnownSpeakerList(
        speakers=[
            KnownSpeakerOut(
                id=s.id,
                name=s.name,
                model_name=s.model_name,
                dimension=s.dimension,
                created_at=s.created_at,
                legacy_embedding=_is_legacy_pickle_blob(s.embedding),
            )
            for s in speakers
        ]
    )


@router.delete("/speakers/known/{speaker_id}", status_code=204)
async def delete_known_speaker(
    speaker_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    spk = await db.get(KnownSpeaker, speaker_id)
    if spk is None:
        raise HTTPException(404, "Known speaker not found")
    await db.delete(spk)
    await db.commit()


@router.post("/jobs/{job_id}/match-speakers", response_model=SpeakerMatchResponse)
async def match_speakers(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> SpeakerMatchResponse:
    """Manually trigger speaker matching for a completed job."""
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

    await _auto_match_speakers(job_id, db)

    # Return the match results
    mappings = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )

    known_list = (await db.execute(select(KnownSpeaker))).scalars().all()
    known_names = {s.name for s in known_list}

    matches = []
    for m in mappings:
        if m.custom_name in known_names:
            matches.append(
                SpeakerMatchResult(
                    speaker_label=m.speaker_label,
                    matched_name=m.custom_name,
                    confidence=None,
                )
            )
        else:
            matches.append(SpeakerMatchResult(speaker_label=m.speaker_label))

    return SpeakerMatchResponse(job_id=job_id, matches=matches)


@router.get("/jobs/{job_id}/speaker-embeddings", response_model=list[SpeakerEmbeddingOut])
async def get_speaker_embeddings(
    job_id: str,
    db: AsyncSession = Depends(get_db),
) -> list[SpeakerEmbeddingOut]:
    embeddings = (
        (await db.execute(select(SpeakerEmbedding).where(SpeakerEmbedding.job_id == job_id)))
        .scalars()
        .all()
    )
    return [
        SpeakerEmbeddingOut(
            speaker_label=e.speaker_label, dimension=e.dimension, model_name=e.model_name
        )
        for e in embeddings
    ]

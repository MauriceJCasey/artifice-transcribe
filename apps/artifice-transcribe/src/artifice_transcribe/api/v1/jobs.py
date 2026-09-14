# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from artifice_output import ProjectLayout, slugify
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from artifice_transcribe.config import settings
from artifice_transcribe.db.models import (
    JobStatus,
    PersistentDictionary,
    SegmentEditVersion,
    SpeakerMapping,
    TranscriptionJob,
    TranscriptSegment,
)
from artifice_transcribe.db.session import get_db
from artifice_transcribe.schemas.transcription import (
    DictionaryResponse,
    DictionaryUpdate,
    EditHistoryResponse,
    EditVersionOut,
    ExportFormat,
    JobMetadataUpdate,
    JobStatusResponse,
    SearchMatch,
    SearchResults,
    SegmentMergeResponse,
    SegmentOut,
    SegmentSplitRequest,
    SegmentSplitResponse,
    SegmentTagUpdate,
    SegmentUpdateRequest,
    SegmentUpdateResponse,
    SpeakerMappingOut,
    SpeakerMapResponse,
    SpeakerRenameRequest,
    TranscriptResponse,
)

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.get("/jobs", response_model=list[JobStatusResponse])
async def list_jobs(db: AsyncSession = Depends(get_db)) -> list[TranscriptionJob]:
    jobs = (
        (await db.execute(select(TranscriptionJob).order_by(TranscriptionJob.created_at.desc())))
        .scalars()
        .all()
    )
    return list(jobs)


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: AsyncSession = Depends(get_db)) -> TranscriptionJob:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    return job


@router.get("/jobs/{job_id}/audio")
async def get_job_audio(job_id: str, db: AsyncSession = Depends(get_db)) -> FileResponse:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    matches = list(settings.upload_path.glob(f"{job_id}_*"))
    if not matches:
        raise HTTPException(404, "Audio file not found")

    return FileResponse(matches[0], filename=job.filename)


@router.get("/jobs/{job_id}/transcript", response_model=TranscriptResponse)
async def get_transcript(job_id: str, db: AsyncSession = Depends(get_db)) -> TranscriptResponse:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

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

    name_map_row = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )
    name_map = {m.speaker_label: m.custom_name for m in name_map_row}

    def _parse_tags(seg: TranscriptSegment) -> list[str]:
        if seg.tags:
            try:
                return json.loads(seg.tags)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    return TranscriptResponse(
        job_id=job_id,
        segments=[
            SegmentOut(
                id=s.id,
                speaker_label=name_map.get(s.speaker_label, s.speaker_label),
                start_time=s.start_time,
                end_time=s.end_time,
                text=s.text,
                tags=_parse_tags(s),
            )
            for s in segs
        ],
    )


@router.patch("/jobs/{job_id}/metadata", response_model=JobStatusResponse)
async def update_job_metadata(
    job_id: str,
    body: JobMetadataUpdate,
    db: AsyncSession = Depends(get_db),
) -> TranscriptionJob:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    return job


@router.patch("/jobs/{job_id}/segments", response_model=SegmentUpdateResponse)
async def update_segments(
    job_id: str,
    body: SegmentUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> SegmentUpdateResponse:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

    from artifice_transcribe.db.models import TranscriptSegment as TS

    updated = 0
    for item in body.updates:
        seg_id = item.get("segment_id")
        text = item.get("text")
        if not seg_id or text is None:
            continue
        seg = await db.get(TS, seg_id)
        if seg and seg.job_id == job_id and seg.text != text:
            db.add(
                SegmentEditVersion(
                    segment_id=seg.id,
                    job_id=job_id,
                    text_before=seg.text,
                    text_after=text,
                )
            )
            seg.text = text
            updated += 1

    await db.commit()
    return SegmentUpdateResponse(job_id=job_id, updated_count=updated)


@router.get("/jobs/{job_id}/segments/{segment_id}/history", response_model=EditHistoryResponse)
async def get_segment_history(
    job_id: str,
    segment_id: str,
    db: AsyncSession = Depends(get_db),
) -> EditHistoryResponse:
    seg = await db.get(TranscriptSegment, segment_id)
    if seg is None or seg.job_id != job_id:
        raise HTTPException(404, "Segment not found")

    versions = (
        (
            await db.execute(
                select(SegmentEditVersion)
                .where(SegmentEditVersion.segment_id == segment_id)
                .order_by(SegmentEditVersion.edited_at.desc())
            )
        )
        .scalars()
        .all()
    )

    return EditHistoryResponse(
        segment_id=segment_id,
        versions=[
            EditVersionOut(
                id=v.id,
                segment_id=v.segment_id,
                text_before=v.text_before,
                text_after=v.text_after,
                edited_at=v.edited_at,
            )
            for v in versions
        ],
    )


@router.patch("/jobs/{job_id}/segments/{segment_id}/tags")
async def update_segment_tags(
    job_id: str,
    segment_id: str,
    body: SegmentTagUpdate,
    db: AsyncSession = Depends(get_db),
) -> dict:
    seg = await db.get(TranscriptSegment, segment_id)
    if seg is None or seg.job_id != job_id:
        raise HTTPException(404, "Segment not found")

    seg.tags = json.dumps(body.tags)
    await db.commit()
    return {"tags": body.tags}


@router.post("/jobs/{job_id}/segments/{segment_id}/split", response_model=SegmentSplitResponse)
async def split_segment(
    job_id: str,
    segment_id: str,
    body: SegmentSplitRequest,
    db: AsyncSession = Depends(get_db),
) -> SegmentSplitResponse:
    seg = await db.get(TranscriptSegment, segment_id)
    if seg is None or seg.job_id != job_id:
        raise HTTPException(404, "Segment not found")

    text = seg.text
    pos = body.split_position
    if pos <= 0 or pos >= len(text):
        raise HTTPException(400, "Split position must be inside the text")

    first_text = text[:pos].strip()
    second_text = text[pos:].strip()
    if not first_text or not second_text:
        raise HTTPException(400, "Split would create an empty segment")

    split_ratio = pos / len(text)
    orig_duration = seg.end_time - seg.start_time
    mid_time = seg.start_time + orig_duration * split_ratio

    seg.text = first_text
    old_end = seg.end_time
    seg.end_time = mid_time

    new_seg = TranscriptSegment(
        job_id=job_id,
        speaker_label=seg.speaker_label,
        start_time=mid_time,
        end_time=old_end,
        text=second_text,
    )
    db.add(new_seg)
    await db.commit()

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

    name_map_row = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )
    name_map = {m.speaker_label: m.custom_name for m in name_map_row}

    def _parse_tags(s: TranscriptSegment) -> list[str]:
        if s.tags:
            try:
                return json.loads(s.tags)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    return SegmentSplitResponse(
        segments=[
            SegmentOut(
                id=s.id,
                speaker_label=name_map.get(s.speaker_label, s.speaker_label),
                start_time=s.start_time,
                end_time=s.end_time,
                text=s.text,
                tags=_parse_tags(s),
            )
            for s in segs
        ],
    )


@router.post("/jobs/{job_id}/segments/{segment_id}/merge", response_model=SegmentMergeResponse)
async def merge_segments(
    job_id: str,
    segment_id: str,
    db: AsyncSession = Depends(get_db),
) -> SegmentMergeResponse:
    seg = await db.get(TranscriptSegment, segment_id)
    if seg is None or seg.job_id != job_id:
        raise HTTPException(404, "Segment not found")

    next_seg = (
        (
            await db.execute(
                select(TranscriptSegment)
                .where(
                    TranscriptSegment.job_id == job_id,
                    TranscriptSegment.start_time > seg.start_time,
                )
                .order_by(TranscriptSegment.start_time)
                .limit(1)
            )
        )
        .scalars()
        .first()
    )

    if next_seg is None:
        raise HTTPException(400, "No next segment to merge with")

    seg.text = seg.text.rstrip(" ") + " " + next_seg.text.lstrip(" ")
    seg.end_time = next_seg.end_time
    deleted_id = next_seg.id
    await db.delete(next_seg)
    await db.commit()

    name_map_row = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )
    name_map = {m.speaker_label: m.custom_name for m in name_map_row}

    def _parse_tags(s: TranscriptSegment) -> list[str]:
        if s.tags:
            try:
                return json.loads(s.tags)
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    return SegmentMergeResponse(
        segment=SegmentOut(
            id=seg.id,
            speaker_label=name_map.get(seg.speaker_label, seg.speaker_label),
            start_time=seg.start_time,
            end_time=seg.end_time,
            text=seg.text,
            tags=_parse_tags(seg),
        ),
        deleted_segment_id=deleted_id,
    )


@router.get("/jobs/{job_id}/speakers", response_model=SpeakerMapResponse)
async def get_speakers(job_id: str, db: AsyncSession = Depends(get_db)) -> SpeakerMapResponse:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    mappings = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )
    return SpeakerMapResponse(
        job_id=job_id,
        speakers=[
            SpeakerMappingOut(speaker_label=m.speaker_label, custom_name=m.custom_name)
            for m in mappings
        ],
    )


@router.patch("/jobs/{job_id}/speakers", response_model=SpeakerMapResponse)
async def rename_speakers(
    job_id: str,
    body: SpeakerRenameRequest,
    db: AsyncSession = Depends(get_db),
) -> SpeakerMapResponse:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    for rename in body.speakers:
        result = await db.execute(
            select(SpeakerMapping).where(
                SpeakerMapping.job_id == job_id,
                SpeakerMapping.speaker_label == rename.speaker_label,
            )
        )
        mapping = result.scalar_one_or_none()
        if mapping:
            mapping.custom_name = rename.custom_name
        else:
            db.add(
                SpeakerMapping(
                    job_id=job_id,
                    speaker_label=rename.speaker_label,
                    custom_name=rename.custom_name,
                )
            )

    await db.commit()

    all_mappings = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )

    return SpeakerMapResponse(
        job_id=job_id,
        speakers=[
            SpeakerMappingOut(speaker_label=m.speaker_label, custom_name=m.custom_name)
            for m in all_mappings
        ],
    )


@router.get("/search", response_model=SearchResults)
async def search_transcripts(
    q: str,
    db: AsyncSession = Depends(get_db),
) -> SearchResults:
    """Full-text search across all completed transcripts."""
    if not q.strip():
        return SearchResults(results=[], total=0)

    search_term = f"%{q.strip()}%"

    segs = (
        (
            await db.execute(
                select(TranscriptSegment)
                .join(TranscriptionJob)
                .where(
                    TranscriptionJob.status == JobStatus.completed,
                    TranscriptSegment.text.ilike(search_term),
                )
                .order_by(TranscriptSegment.start_time)
                .limit(100)
            )
        )
        .scalars()
        .all()
    )

    results = []
    for seg in segs:
        job = await db.get(TranscriptionJob, seg.job_id)
        if job is None:
            continue
        results.append(
            SearchMatch(
                job_id=seg.job_id,
                filename=job.filename,
                segment_id=seg.id,
                speaker_label=seg.speaker_label,
                text=seg.text,
                start_time=seg.start_time,
                end_time=seg.end_time,
                interviewee=job.interviewee,
                interviewer=job.interviewer,
                interview_date=job.interview_date,
                project_name=job.project_name,
            )
        )

    return SearchResults(results=results, total=len(results))


@router.get("/jobs/{job_id}/export")
async def export_transcript(
    job_id: str,
    format: ExportFormat = ExportFormat.json,
    db: AsyncSession = Depends(get_db),
) -> Response:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    if job.status != JobStatus.completed:
        raise HTTPException(409, f"Job is {job.status.value}, not completed")

    from artifice_transcribe.services import exports

    content_type_map = {
        ExportFormat.json: "application/json",
        ExportFormat.srt: "text/srt",
        ExportFormat.vtt: "text/vtt",
        ExportFormat.txt: "text/plain",
        ExportFormat.md: "text/markdown",
        ExportFormat.pdf: "application/pdf",
        ExportFormat.ohms: "application/xml",
        ExportFormat.tei: "application/xml",
    }
    exporters = {
        ExportFormat.json: exports.export_json,
        ExportFormat.srt: exports.export_srt,
        ExportFormat.vtt: exports.export_vtt,
        ExportFormat.txt: exports.export_txt,
        ExportFormat.md: exports.export_md,
        ExportFormat.pdf: exports.export_pdf,
        ExportFormat.ohms: exports.export_ohms,
        ExportFormat.tei: exports.export_tei,
    }

    body = await exporters[format](db, job_id)
    is_binary = format in (ExportFormat.pdf,)
    content = body if is_binary or not isinstance(body, str) else body.encode("utf-8")
    try:
        project = ProjectLayout(
            settings.output_path,
            job.project_name or Path(job.filename).stem or "transcript",
            create=True,
        )
        export_dir = project.export_dir("transcript")
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / f"{slugify(Path(job.filename).stem)}.{format.value}").write_bytes(content)
    except OSError:
        # Preserve the HTTP download if the optional local archive is unavailable.
        pass
    return Response(
        content=content,
        media_type=content_type_map[format],
        headers={
            "Content-Disposition": f'attachment; filename="transcript_{job_id}.{format.value}"'
        },
    )


# ── Persistent Dictionary ──────────────────────────────────────────────


@router.get("/dictionary", response_model=DictionaryResponse | None)
async def get_dictionary(db: AsyncSession = Depends(get_db)) -> DictionaryResponse | None:
    row = (await db.execute(select(PersistentDictionary).limit(1))).scalars().first()
    if row is None:
        return None
    return DictionaryResponse(id=row.id, words=row.words, updated_at=row.updated_at)


@router.put("/dictionary", response_model=DictionaryResponse)
async def update_dictionary(
    body: DictionaryUpdate,
    db: AsyncSession = Depends(get_db),
) -> DictionaryResponse:
    row = (await db.execute(select(PersistentDictionary).limit(1))).scalars().first()
    if row is None:
        row = PersistentDictionary(words=body.words)
        db.add(row)
    else:
        row.words = body.words
        row.updated_at = datetime.now(UTC)
    await db.commit()
    return DictionaryResponse(id=row.id, words=row.words, updated_at=row.updated_at)


@router.delete("/jobs/{job_id}", status_code=204)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)) -> None:
    job = await db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(404, "Job not found")

    upload_dir = settings.upload_path
    for p in upload_dir.glob(f"{job_id}_*"):
        p.unlink(missing_ok=True)

    await db.delete(job)
    await db.commit()

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
from io import StringIO

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from artifice_transcribe.db.models import SpeakerMapping, TranscriptionJob, TranscriptSegment


def _fmt_ts_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_ts_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _fmt_ts_ohms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _load_data(
    db: AsyncSession, job_id: str
) -> tuple[list[TranscriptSegment], dict[str, str]]:
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

    mappings = (
        (await db.execute(select(SpeakerMapping).where(SpeakerMapping.job_id == job_id)))
        .scalars()
        .all()
    )
    name_map = {m.speaker_label: m.custom_name for m in mappings}

    return list(segs), name_map


def _speaker_name(label: str, name_map: dict[str, str]) -> str:
    return name_map.get(label, label)


async def export_json(db: AsyncSession, job_id: str) -> str:
    segs, name_map = await _load_data(db, job_id)
    data = [
        {
            "speaker": _speaker_name(s.speaker_label, name_map),
            "speaker_label": s.speaker_label,
            "start": round(s.start_time, 3),
            "end": round(s.end_time, 3),
            "text": s.text,
        }
        for s in segs
    ]
    return json.dumps({"job_id": job_id, "segments": data}, indent=2, ensure_ascii=False)


async def export_srt(db: AsyncSession, job_id: str) -> str:
    segs, name_map = await _load_data(db, job_id)
    buf = StringIO()
    for i, s in enumerate(segs, 1):
        buf.write(f"{i}\n")
        buf.write(f"{_fmt_ts_srt(s.start_time)} --> {_fmt_ts_srt(s.end_time)}\n")
        buf.write(f"[{_speaker_name(s.speaker_label, name_map)}] {s.text}\n\n")
    return buf.getvalue()


async def export_vtt(db: AsyncSession, job_id: str) -> str:
    segs, name_map = await _load_data(db, job_id)
    buf = StringIO()
    buf.write("WEBVTT\n\n")
    for i, s in enumerate(segs, 1):
        buf.write(f"{i}\n")
        buf.write(f"{_fmt_ts_vtt(s.start_time)} --> {_fmt_ts_vtt(s.end_time)}\n")
        buf.write(f"[{_speaker_name(s.speaker_label, name_map)}] {s.text}\n\n")
    return buf.getvalue()


async def export_txt(db: AsyncSession, job_id: str) -> str:
    segs, name_map = await _load_data(db, job_id)
    buf = StringIO()
    current_speaker = None
    for s in segs:
        name = _speaker_name(s.speaker_label, name_map)
        if name != current_speaker:
            current_speaker = name
            buf.write(f"\n[{name}]\n")
        buf.write(f"{s.text} ")
    return buf.getvalue().strip() + "\n"


async def export_md(db: AsyncSession, job_id: str) -> str:
    segs, name_map = await _load_data(db, job_id)
    buf = StringIO()
    buf.write(f"# Transcript — {job_id}\n\n")
    current_speaker = None
    for s in segs:
        name = _speaker_name(s.speaker_label, name_map)
        if name != current_speaker:
            current_speaker = name
            buf.write(f"\n**{name}**\n\n")
        buf.write(f"{s.text}\n\n")
    return buf.getvalue()


def _sanitize_pdf(text: str) -> str:
    """Replace common Unicode characters with Latin-1 fallbacks and strip
    any remaining characters outside the core-font supported range."""
    subs = {
        "\u2014": "--",
        "\u2013": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u00a0": " ",
    }
    for old, new in subs.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


async def export_pdf(db: AsyncSession, job_id: str) -> bytes:
    from fpdf import FPDF

    segs, name_map = await _load_data(db, job_id)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _sanitize_pdf(f"Transcript \u2014 {job_id}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    current_speaker = None
    for s in segs:
        name = _speaker_name(s.speaker_label, name_map)
        if name != current_speaker:
            current_speaker = name
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 7, _sanitize_pdf(name), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 5, _sanitize_pdf(s.text))
        pdf.ln(2)

    return bytes(pdf.output())


async def export_ohms(db: AsyncSession, job_id: str) -> str:
    """OHMS XML export (Oral History Metadata Synchronizer format)."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    job = await db.get(TranscriptionJob, job_id)
    segs, name_map = await _load_data(db, job_id)

    root = ET.Element("ROOT")
    record = ET.SubElement(root, "record")

    ET.SubElement(record, "id").text = job_id
    ET.SubElement(record, "title").text = job.filename if job else job_id
    ET.SubElement(record, "interviewee").text = job.interviewee or ""
    ET.SubElement(record, "interviewer").text = job.interviewer or ""
    ET.SubElement(record, "date").text = job.interview_date or ""
    ET.SubElement(record, "location").text = job.location or ""
    ET.SubElement(record, "collection_id").text = job.collection_id or ""
    ET.SubElement(record, "project_name").text = job.project_name or ""
    ET.SubElement(record, "access_restrictions").text = job.access_restrictions or ""

    transcript = ET.SubElement(record, "transcript")
    for s in segs:
        sync = ET.SubElement(transcript, "sync")
        sync.set("time", _fmt_ts_ohms(s.start_time))
        speaker = _speaker_name(s.speaker_label, name_map)
        sync.text = f"[{speaker}] {s.text}"

    rough_raw = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough_raw)
    return dom.toprettyxml(indent="  ")


async def export_tei(db: AsyncSession, job_id: str) -> str:
    """TEI XML export (Text Encoding Initiative standard)."""
    import xml.etree.ElementTree as ET
    from xml.dom import minidom

    job = await db.get(TranscriptionJob, job_id)
    segs, name_map = await _load_data(db, job_id)

    TEI_NS = "http://www.tei-c.org/ns/1.0"

    ET.register_namespace("", TEI_NS)
    root = ET.Element(f"{{{TEI_NS}}}TEI")
    teiHeader = ET.SubElement(root, f"{{{TEI_NS}}}teiHeader")
    fileDesc = ET.SubElement(teiHeader, f"{{{TEI_NS}}}fileDesc")
    titleStmt = ET.SubElement(fileDesc, f"{{{TEI_NS}}}titleStmt")
    ET.SubElement(titleStmt, f"{{{TEI_NS}}}title").text = (
        f"Oral History Transcript: {job.filename}" if job else job_id
    )

    if job and job.interviewee:
        author = ET.SubElement(titleStmt, f"{{{TEI_NS}}}author")
        ET.SubElement(author, f"{{{TEI_NS}}}name").text = job.interviewee
    if job and job.interviewer:
        respStmt = ET.SubElement(titleStmt, f"{{{TEI_NS}}}respStmt")
        ET.SubElement(respStmt, f"{{{TEI_NS}}}resp").text = "Interviewer"
        ET.SubElement(respStmt, f"{{{TEI_NS}}}name").text = job.interviewer

    publicationStmt = ET.SubElement(fileDesc, f"{{{TEI_NS}}}publicationStmt")
    ET.SubElement(publicationStmt, f"{{{TEI_NS}}}p").text = "Generated by ArtificeTranscribe"

    sourceDesc = ET.SubElement(fileDesc, f"{{{TEI_NS}}}sourceDesc")
    recordingStmt = ET.SubElement(sourceDesc, f"{{{TEI_NS}}}recordingStmt")
    ET.SubElement(recordingStmt, f"{{{TEI_NS}}}p").text = job.filename if job else job_id

    if job and job.interview_date:
        ET.SubElement(sourceDesc, f"{{{TEI_NS}}}p").text = f"Date: {job.interview_date}"
    if job and job.location:
        ET.SubElement(sourceDesc, f"{{{TEI_NS}}}p").text = f"Location: {job.location}"

    text = ET.SubElement(root, f"{{{TEI_NS}}}text")
    body = ET.SubElement(text, f"{{{TEI_NS}}}body")
    div = ET.SubElement(body, f"{{{TEI_NS}}}div")
    ET.SubElement(div, f"{{{TEI_NS}}}head").text = "Transcript"

    current_speaker = None
    for s in segs:
        name = _speaker_name(s.speaker_label, name_map)
        if name != current_speaker:
            current_speaker = name
            sp = ET.SubElement(div, f"{{{TEI_NS}}}sp")
            ET.SubElement(sp, f"{{{TEI_NS}}}speaker").text = name
            ET.SubElement(sp, f"{{{TEI_NS}}}ab", {"type": "speech"}).text = s.text
        else:
            sps = div.findall(f"{{{TEI_NS}}}sp")
            sp = sps[-1] if sps else None
            if sp is not None:
                ET.SubElement(sp, f"{{{TEI_NS}}}ab", {"type": "speech"}).text = s.text

    rough_raw = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(rough_raw)
    return dom.toprettyxml(indent="  ")

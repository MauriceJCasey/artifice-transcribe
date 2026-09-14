# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Enum, Float, ForeignKey, LargeBinary, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from artifice_transcribe.schemas.transcription import JobStatus


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


# ── Embedding serialisation helpers ────────────────────────────────────────
#
# Speaker embeddings are stored as raw float32 bytes (ndarray.tobytes())
# so that no consumer ever deserialises arbitrary objects from the database.
# Both KnownSpeaker.embedding and SpeakerEmbedding.embedding use these.


def pack_embedding(embedding: "np.ndarray") -> bytes:
    """Cast *embedding* to ``np.float32`` and return its raw bytes."""
    import numpy as np

    return np.asarray(embedding, dtype=np.float32).tobytes()


class LegacyEmbeddingError(ValueError):
    """Raised when a stored embedding blob is a pickle payload predating
    the raw-float32 format.  The speaker must be re-enrolled."""


# Pickle protocol 2+ frames the payload as PROTO (``\x80``) + protocol byte.
# Protocol 5 has been the default since Python 3.8, so a legacy row written by
# any Python this project has ever supported carries one of these.
_PICKLE_PROTOCOLS = frozenset({2, 3, 4, 5})

# Every well-formed pickle ends with the STOP opcode.
_PICKLE_STOP = 0x2E  # b"."


def _is_legacy_pickle_blob(blob: bytes) -> bool:
    """Return *True* if *blob* looks like a pickled Python object.

    Pickle protocol 2+ payloads begin with the opcode ``\\x80`` followed by
    a protocol byte, and end with the STOP opcode ``.``.  Protocol 0/1
    payloads start with a printable ASCII opcode (e.g. ``(``, ``i``) and are
    harder to detect without a full parse, but in practice every pickle
    produced by a modern Python has been protocol 2+ since 3.8 raised the
    default.

    All three bytes are checked, not just the leading ``\\x80``.  A raw
    float32 vector is arbitrary binary, so roughly one embedding in 256 has
    a leading byte of ``0x80`` purely by chance — under the prefix-only test
    those rows were refused as "legacy" and the user was told to re-enrol a
    speaker whose data was perfectly readable.  CI caught exactly that on a
    random 512-dim vector.  Requiring the protocol byte and the trailing
    STOP as well takes the collision rate from ~1 in 256 to ~1 in 4 million
    while still matching every real pickle.

    The check remains a pure byte inspection; ``pickle.loads`` is never
    called.
    """
    return (
        len(blob) >= 3
        and blob[0] == 0x80
        and blob[1] in _PICKLE_PROTOCOLS
        and blob[-1] == _PICKLE_STOP
    )


def unpack_embedding(blob: bytes, dimension: int | None = None) -> "np.ndarray":
    """Return ``np.frombuffer(blob, dtype=np.float32)`` after validating
    that *blob* is a whole number of float32 values and, when *dimension*
    is given, that the vector length matches."""
    if _is_legacy_pickle_blob(blob):
        raise LegacyEmbeddingError(
            "This embedding blob predates the raw-float32 format and "
            "cannot be read. The speaker must be re-enrolled."
        )
    if len(blob) == 0:
        raise ValueError("Embedding blob is empty")
    if len(blob) % 4 != 0:
        raise ValueError(f"Embedding blob is {len(blob)} bytes, not a multiple of 4 (float32)")
    actual_dim = len(blob) // 4
    if dimension is not None and actual_dim != dimension:
        raise ValueError(f"Embedding blob has dimension {actual_dim}, expected {dimension}")
    import numpy as np

    return np.frombuffer(blob, dtype=np.float32)


class TranscriptionJob(Base):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    progress_percentage: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    options: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string

    # Oral history metadata
    interviewee: Mapped[str | None] = mapped_column(String(256), nullable=True)
    interviewer: Mapped[str | None] = mapped_column(String(256), nullable=True)
    interview_date: Mapped[str | None] = mapped_column(String(64), nullable=True)
    location: Mapped[str | None] = mapped_column(String(256), nullable=True)
    project_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    collection_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    access_restrictions: Mapped[str | None] = mapped_column(String(512), nullable=True)
    custom_vocabulary: Mapped[str | None] = mapped_column(Text, nullable=True)  # comma-separated

    # passive_deletes=True: TranscriptSegment.job_id and SpeakerMapping.job_id
    # both declare ForeignKey(..., ondelete="CASCADE"), and db/session.py
    # unconditionally enables PRAGMA foreign_keys=ON, so SQLite itself already
    # cascades a job's deletion down to its segments and speaker mappings.
    # Without passive_deletes=True, SQLAlchemy doesn't trust the DB to do this
    # and instead SELECTs every child row into memory first so it can stage
    # each one for individual deletion -- for a long oral-history interview
    # that means loading hundreds to low-thousands of Text-column transcript
    # rows just to delete a job. Do not remove this: it looks redundant with
    # the FK's ON DELETE CASCADE but is what stops SQLAlchemy from bypassing it.
    segments: Mapped[list[TranscriptSegment]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )
    speakers: Mapped[list[SpeakerMapping]] = relationship(
        back_populates="job", cascade="all, delete-orphan", passive_deletes=True
    )


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("transcription_jobs.id", ondelete="CASCADE"), index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(32))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON list of tag strings

    job: Mapped[TranscriptionJob] = relationship(back_populates="segments")


class SpeakerMapping(Base):
    __tablename__ = "speaker_mappings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    job_id: Mapped[str] = mapped_column(
        ForeignKey("transcription_jobs.id", ondelete="CASCADE"), index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(32))
    custom_name: Mapped[str] = mapped_column(String(128))

    job: Mapped[TranscriptionJob] = relationship(back_populates="speakers")


class PersistentDictionary(Base):
    """A single-row table holding a comma-separated global vocabulary list
    that is merged into every transcription job's hotwords."""

    __tablename__ = "persistent_dictionary"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    words: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class KnownSpeaker(Base):
    """An enrolled speaker whose voice embedding is stored for cross-session
    recognition."""

    __tablename__ = "known_speakers"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(256))
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # raw float32 bytes (pack_embedding)
    model_name: Mapped[str] = mapped_column(String(64), default="pyannote/embedding")
    dimension: Mapped[int] = mapped_column(default=512)
    sample_audio_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class SpeakerEmbedding(Base):
    """Per-speaker centroid embedding for a specific job, used for cross-session
    speaker matching."""

    __tablename__ = "speaker_embeddings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    # ondelete="CASCADE": this row is per-job derived data (a speaker
    # centroid for cross-session matching) with no meaning once its job is
    # gone. Before this, job_id was a plain unconstrained column -- deleting
    # a job never cleaned these up, so they accumulated forever. No ORM
    # relationship is declared on TranscriptionJob for this: nothing in the
    # codebase accesses speaker embeddings as a collection off a job, and
    # PRAGMA foreign_keys=ON (db/session.py) is enough for SQLite's own
    # ON DELETE CASCADE to handle cleanup with no ORM involvement at all --
    # the same reasoning that already makes SegmentEditVersion's cleanup
    # work transitively through its own segment_id FK, verified empirically
    # rather than assumed.
    job_id: Mapped[str] = mapped_column(
        ForeignKey("transcription_jobs.id", ondelete="CASCADE"), index=True
    )
    speaker_label: Mapped[str] = mapped_column(String(32))
    embedding: Mapped[bytes] = mapped_column(LargeBinary)  # raw float32 bytes (pack_embedding)
    model_name: Mapped[str] = mapped_column(String(64), default="pyannote/embedding")
    dimension: Mapped[int] = mapped_column(default=512)


class SegmentEditVersion(Base):
    __tablename__ = "segment_edit_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    segment_id: Mapped[str] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE"), index=True
    )
    # Denormalized for fast job_id-scoped lookups only -- deliberately not
    # its own ForeignKey. segment_id above already cascades from
    # transcript_segments.id, which itself cascades from
    # transcription_jobs.id, so deleting a job already deletes this row
    # transitively through that two-hop path (verified empirically: SQLite
    # honours ON DELETE CASCADE across multiple FK hops in one statement
    # when PRAGMA foreign_keys=ON, which db/session.py always sets). Adding
    # a second FK here would be redundant, not a fix for anything.
    job_id: Mapped[str] = mapped_column(String(32), index=True)
    text_before: Mapped[str] = mapped_column(Text)
    text_after: Mapped[str] = mapped_column(Text)
    edited_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

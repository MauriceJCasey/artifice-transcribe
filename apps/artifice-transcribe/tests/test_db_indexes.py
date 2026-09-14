# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression coverage for the five job_id/segment_id indexes added to
``db/models.py`` (TranscriptSegment.job_id, SpeakerMapping.job_id,
SpeakerEmbedding.job_id, SegmentEditVersion.segment_id,
SegmentEditVersion.job_id).

Nearly every read path filters by ``job_id`` (job detail, segment listing,
speaker mappings, exports, edit history); without an index these degrade to
full table scans as history grows. This module guards two things:

1. A freshly created database carries all five indexes, under the exact
   names SQLAlchemy's default ``ix_<table>_<column>`` convention assigns.
2. An *already existing* database file — one created before these indexes
   were declared, matching what a running deployment's SQLite file looks
   like today — gets the indexes retrofitted by the app's startup
   (``lifespan``) rather than silently staying un-indexed. This is not
   automatic: ``Base.metadata.create_all`` uses ``checkfirst=True``, which
   skips DDL entirely for a table that already exists on disk, so a newly
   declared ``index=True`` is never applied to that table by ``create_all``
   alone. Verified empirically (see PR description); ``lifespan`` now
   issues explicit ``CREATE INDEX IF NOT EXISTS`` statements after
   ``create_all`` to cover exactly this case.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from artifice_transcribe.db.models import (
    Base,
    SegmentEditVersion,
    SpeakerEmbedding,
    SpeakerMapping,
    TranscriptionJob,
    TranscriptSegment,
)
from sqlalchemy import ForeignKey, String, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# The five indexes this fix adds, named per SQLAlchemy's default
# ``ix_<table>_<column>`` convention — confirmed empirically against a
# freshly created database, not assumed.
_EXPECTED_INDEXES = {
    "ix_transcript_segments_job_id",
    "ix_speaker_mappings_job_id",
    "ix_speaker_embeddings_job_id",
    "ix_segment_edit_versions_segment_id",
    "ix_segment_edit_versions_job_id",
}


def _index_names(db_path: Path) -> set[str]:
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        con.close()
    return {row[0] for row in rows}


@pytest.mark.asyncio
async def test_fresh_database_has_all_five_indexes(tmp_path):
    """A brand-new database, built straight from ``Base.metadata``, carries
    all five indexes under the expected names."""
    db_path = tmp_path / "fresh.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()

    found = _index_names(db_path)
    missing = _EXPECTED_INDEXES - found
    assert not missing, f"missing indexes on a freshly created database: {missing}"


# ---------------------------------------------------------------------------
# Retrofit coverage: simulate a database created *before* these indexes were
# declared (an already-running deployment's SQLite file), then prove the
# app's startup path actually adds them.
# ---------------------------------------------------------------------------


class _OldBase(DeclarativeBase):
    """Mirrors the pre-fix schema: the same five tables/columns, but with no
    ``index=True`` on any of the columns this fix targets."""


class _OldJob(_OldBase):
    __tablename__ = "transcription_jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)


class _OldSegment(_OldBase):
    __tablename__ = "transcript_segments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("transcription_jobs.id", ondelete="CASCADE"))


class _OldSpeakerMapping(_OldBase):
    __tablename__ = "speaker_mappings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("transcription_jobs.id", ondelete="CASCADE"))


class _OldSpeakerEmbedding(_OldBase):
    __tablename__ = "speaker_embeddings"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    job_id: Mapped[str] = mapped_column(String(32))


class _OldSegmentEditVersion(_OldBase):
    __tablename__ = "segment_edit_versions"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    segment_id: Mapped[str] = mapped_column(
        ForeignKey("transcript_segments.id", ondelete="CASCADE")
    )
    job_id: Mapped[str] = mapped_column(String(32))


@pytest.mark.asyncio
async def test_lifespan_retrofits_indexes_onto_pre_existing_database(tmp_path, monkeypatch):
    """A database created under the *old*, un-indexed schema gets the five
    indexes added when the app starts up, via ``lifespan``'s explicit
    ``CREATE INDEX IF NOT EXISTS`` statements — not just at ``create_all``
    time for brand-new tables."""
    db_path = tmp_path / "existing.db"

    # Phase 1: create the tables under the OLD (un-indexed) schema and
    # close the connection, simulating a database that predates this fix.
    old_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with old_engine.begin() as conn:
            await conn.run_sync(_OldBase.metadata.create_all)
    finally:
        await old_engine.dispose()

    assert _index_names(db_path) == set(), "test setup should start with no indexes"

    # Phase 2: point the app's module-level engine at this pre-existing
    # database and run its startup lifespan, exactly as happens on a real
    # launch against an existing data directory.
    import artifice_transcribe.main as main_module

    new_engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setattr(main_module, "engine", new_engine)
    # ``settings.data_path`` is a read-only property returning this module
    # constant; patch the constant rather than the property itself.
    monkeypatch.setattr("artifice_transcribe.config._USER_DATA_PATH", tmp_path)

    try:
        async with main_module.lifespan(main_module.app):
            pass
    finally:
        await new_engine.dispose()

    found = _index_names(db_path)
    missing = _EXPECTED_INDEXES - found
    assert not missing, f"lifespan failed to retrofit indexes onto an existing database: {missing}"


# ---------------------------------------------------------------------------
# Cascade-delete coverage: TranscriptionJob.segments/.speakers declare
# cascade="all, delete-orphan" *and* passive_deletes=True. The two child
# foreign keys already declare ondelete="CASCADE", and PRAGMA foreign_keys=ON
# is enabled on every connection, so the database itself is fully able to
# cascade a job deletion down to its segments and speaker mappings.
# passive_deletes=True tells SQLAlchemy to trust that and skip loading every
# child row into memory first. This section proves both that the delete
# still fully cleans up (correctness) and that no SELECT against either
# child table is issued as part of it (the actual behavior the fix changes).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_delete_cascades_to_segments_and_speakers(tmp_path):
    """Deleting a TranscriptionJob via ``db.delete(job); db.commit()`` (the
    same pattern ``delete_job`` in api/v1/jobs.py uses) leaves zero rows
    behind in transcript_segments/speaker_mappings for that job, and does so
    without SQLAlchemy ever SELECTing those child rows into memory -- it
    defers entirely to the database's own ON DELETE CASCADE."""
    db_path = tmp_path / "cascade.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    # Same PRAGMA the app's real engine sets in db/session.py -- without it
    # SQLite would not honor ON DELETE CASCADE at all, and passive_deletes=True
    # would silently orphan the child rows instead of cleaning them up.
    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            job = TranscriptionJob(filename="interview.wav")
            session.add(job)
            await (
                session.flush()
            )  # populate job.id (Python-side default) before children reference it
            for i in range(5):
                session.add(
                    TranscriptSegment(
                        job_id=job.id,
                        speaker_label="A",
                        start_time=float(i),
                        end_time=float(i + 1),
                        text=f"segment {i}",
                    )
                )
            for label in ("A", "B"):
                session.add(
                    SpeakerMapping(
                        job_id=job.id, speaker_label=label, custom_name=f"Speaker {label}"
                    )
                )
            await session.commit()

            job_id = job.id

            # Capture every statement executed during the delete+commit only.
            executed_statements: list[str] = []

            def _capture(_conn, _cursor, statement, _parameters, _context, _executemany):
                executed_statements.append(statement)

            event.listen(engine.sync_engine, "before_cursor_execute", _capture)
            try:
                await session.delete(job)
                await session.commit()
            finally:
                event.remove(engine.sync_engine, "before_cursor_execute", _capture)
    finally:
        await engine.dispose()

    # Correctness: the database's own ON DELETE CASCADE actually cleaned up
    # both child tables, not just the parent row.
    con = sqlite3.connect(str(db_path))
    try:
        seg_count = con.execute(
            "SELECT COUNT(*) FROM transcript_segments WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        spk_count = con.execute(
            "SELECT COUNT(*) FROM speaker_mappings WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        job_count = con.execute(
            "SELECT COUNT(*) FROM transcription_jobs WHERE id=?", (job_id,)
        ).fetchone()[0]
    finally:
        con.close()
    assert seg_count == 0, "transcript_segments rows survived the job delete"
    assert spk_count == 0, "speaker_mappings rows survived the job delete"
    assert job_count == 0, "the job row itself was not deleted"

    # Behavioral: passive_deletes=True means SQLAlchemy never loads the child
    # rows to stage them for individual deletion -- confirm no SELECT against
    # either child table was issued while deleting the job.
    loaded_children = [
        stmt
        for stmt in executed_statements
        if stmt.strip().upper().startswith("SELECT")
        and ("transcript_segments" in stmt.lower() or "speaker_mappings" in stmt.lower())
    ]
    assert not loaded_children, (
        "passive_deletes=True should stop SQLAlchemy from SELECTing child rows "
        f"before the database's ON DELETE CASCADE handles them; found: {loaded_children}"
    )


# ---------------------------------------------------------------------------
# SpeakerEmbedding / SegmentEditVersion cleanup on job delete
# ---------------------------------------------------------------------------
# SpeakerEmbedding.job_id used to be a plain unconstrained column -- deleting
# a job never cleaned these rows up, so they accumulated forever (the
# opposite failure mode from the passive_deletes fix above: missing cleanup,
# not over-eager loading). Now a real ForeignKey(ondelete="CASCADE").
#
# SegmentEditVersion.job_id stays deliberately unconstrained: its own
# segment_id already cascades from transcript_segments.id, which itself
# cascades from transcription_jobs.id, so a job delete already cleans these
# rows up transitively through that two-hop path with no ORM relationship
# involved at all. This test proves both halves empirically rather than
# assuming either.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_job_delete_cleans_up_speaker_embeddings_and_edit_versions(tmp_path):
    """SpeakerEmbedding rows are deleted via their own new FK; SegmentEditVersion
    rows are deleted transitively through segment_id -> transcript_segments.job_id,
    with no FK of their own needed."""
    db_path = tmp_path / "cascade2.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")

    @event.listens_for(engine.sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        async with async_session() as session:
            job = TranscriptionJob(filename="interview.wav")
            session.add(job)
            await session.flush()

            seg = TranscriptSegment(
                job_id=job.id,
                speaker_label="A",
                start_time=0.0,
                end_time=1.0,
                text="hello",
            )
            session.add(seg)
            await session.flush()

            session.add(
                SegmentEditVersion(
                    segment_id=seg.id,
                    job_id=job.id,
                    text_before="hello",
                    text_after="hi",
                )
            )
            session.add(
                SpeakerEmbedding(
                    job_id=job.id,
                    speaker_label="A",
                    embedding=b"\x00" * 8,
                    dimension=2,
                )
            )
            await session.commit()

            job_id = job.id

            await session.delete(job)
            await session.commit()
    finally:
        await engine.dispose()

    con = sqlite3.connect(str(db_path))
    try:
        emb_count = con.execute(
            "SELECT COUNT(*) FROM speaker_embeddings WHERE job_id=?", (job_id,)
        ).fetchone()[0]
        edit_count = con.execute(
            "SELECT COUNT(*) FROM segment_edit_versions WHERE job_id=?", (job_id,)
        ).fetchone()[0]
    finally:
        con.close()

    assert emb_count == 0, "speaker_embeddings rows survived the job delete"
    assert edit_count == 0, (
        "segment_edit_versions rows survived the job delete "
        "(transitive cascade through segment_id did not fire)"
    )

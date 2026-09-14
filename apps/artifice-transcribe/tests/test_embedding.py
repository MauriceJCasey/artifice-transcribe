# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import select

from artifice_transcribe.db.models import (
    JobStatus,
    LegacyEmbeddingError,
    SpeakerMapping,
    pack_embedding,
    unpack_embedding,
)


class TestPackUnpackRoundTrip:
    """A known float vector survives write → read with values intact."""

    def test_round_trip_1d(self):
        original = np.array([0.1, 0.2, 0.3, -0.4, 0.5], dtype=np.float32)
        blob = pack_embedding(original)
        assert isinstance(blob, bytes)
        assert len(blob) == 5 * 4  # 5 float32 = 20 bytes

        restored = unpack_embedding(blob, dimension=5)
        assert restored.dtype == np.float32
        np.testing.assert_array_equal(original, restored)

    def test_round_trip_high_dimensional(self):
        """512-dim is the default dimension used by PyAnnote embeddings."""
        original = np.random.randn(512).astype(np.float32)
        blob = pack_embedding(original)
        assert len(blob) == 512 * 4

        restored = unpack_embedding(blob, dimension=512)
        assert len(restored) == 512
        np.testing.assert_array_almost_equal(original, restored, decimal=5)

    def test_round_trip_2d_collapses_to_1d(self):
        """pack_embedding casts via asarray so a (1,N) shape becomes 1-D."""
        original = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        blob = pack_embedding(original)
        restored = unpack_embedding(blob, dimension=3)
        assert restored.shape == (3,)
        np.testing.assert_array_equal(restored, np.array([0.1, 0.2, 0.3], dtype=np.float32))

    def test_cast_to_float32(self):
        """A float64 input is safely cast to float32."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        blob = pack_embedding(original)
        assert len(blob) == 3 * 4
        restored = unpack_embedding(blob, dimension=3)
        assert restored.dtype == np.float32


class TestUnpackValidation:
    """Malformed or truncated blobs must raise rather than returning a
    silently wrong vector."""

    def test_truncated_blob_raises(self):
        """A blob whose length is not a multiple of 4 is malformed."""
        with pytest.raises(ValueError, match="not a multiple of 4"):
            unpack_embedding(b"\x00\x00\x00", dimension=None)

    def test_wrong_dimension_raises(self):
        """A blob with the wrong number of floats for the declared
        dimension must raise."""
        blob = pack_embedding(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
        with pytest.raises(ValueError, match="expected 5"):
            unpack_embedding(blob, dimension=5)

    def test_empty_blob_raises(self):
        with pytest.raises(ValueError, match="empty"):
            unpack_embedding(b"", dimension=0)

    def test_dimension_none_is_loose(self):
        """When dimension is not provided the check is skipped."""
        blob = pack_embedding(np.array([0.1, 0.2, 0.3], dtype=np.float32))
        restored = unpack_embedding(blob, dimension=None)
        assert len(restored) == 3

    def test_correct_dimension_passes(self):
        """Matching dimension validates cleanly."""
        blob = pack_embedding(np.array([0.1, 0.2], dtype=np.float32))
        restored = unpack_embedding(blob, dimension=2)
        assert len(restored) == 2

    def test_large_dimension_mismatch(self):
        """Big mismatches also raise."""
        blob = pack_embedding(np.zeros(1024, dtype=np.float32))
        with pytest.raises(ValueError, match="expected 512"):
            unpack_embedding(blob, dimension=512)


class TestLegacyPickleDetection:
    """Pre-existing pickled rows must be detected and refused without ever
    calling ``pickle.loads``."""

    # Pickle protocol 2 payload for a 4-element float list:
    #   \x80\x02]q\x00(G?\xd9\x99\x9aG?\xd9\x99\x9aG?\xd9\x99\x9aG?\xd9\x99\x9ae.
    # (length 32 = 8 * 4, so it also exercises the silent-wrong-vector path)
    _PICKLE_FLOATS = b"\x80\x02]q\x00(G?\xd9\x99\x9aG?\xd9\x99\x9aG?\xd9\x99\x9aG?\xd9\x99\x9ae."

    # A pure Python float is pickled via the FLOAT opcode 'F':
    #   \x80\x02F1.0\n.
    # This produces a short blob whose length happens to be a multiple of 4
    # (12 bytes), so it would slip past both the empty and modulo-4 guards
    # if the \x80 check wasn't there.
    _PICKLE_SINGLE_FLOAT = b"\x80\x02F1.0\n."

    def test_pickle_blob_raises_legacy_embedding_error(self):
        """Real pickle bytes must raise LegacyEmbeddingError, not a bare
        ValueError."""
        # The payload is a valid pickle, so the \x80 check fires first.
        with pytest.raises(LegacyEmbeddingError, match="must be re-enrolled"):
            unpack_embedding(self._PICKLE_FLOATS, dimension=8)

    def test_pickle_blob_is_never_unpickled(self):
        """We never call pickle.loads — prove it by checking the exception
        type is our named exception (not the ValueError that a malformed
        float32 blob would give)."""
        with pytest.raises(LegacyEmbeddingError):
            unpack_embedding(self._PICKLE_FLOATS)

    def test_pickle_blob_multiple_of_4_wrong_dimension_still_raises_legacy(self):
        """A pickle blob whose length is a multiple of 4 and also disagrees
        with the stored dimension must raise LegacyEmbeddingError (the \x80
        check fires before dimension validation), closing the
        silent-wrong-vector path."""
        # Test with a dimension that doesn't match the blob length / 4.
        with pytest.raises(LegacyEmbeddingError):
            unpack_embedding(self._PICKLE_FLOATS, dimension=12)

    def test_pickle_blob_multiple_of_4_matching_dimension_still_raises_legacy(self):
        """Even when the pickle blob length / 4 happens to match the stored
        dimension, the \x80 check must fire first.  This is the
        silent-wrong-vector scenario: without the check, we'd return garbage
        floats."""
        matching_dim = len(self._PICKLE_FLOATS) // 4  # 8
        with pytest.raises(LegacyEmbeddingError):
            unpack_embedding(self._PICKLE_FLOATS, dimension=matching_dim)

    def test_short_pickle_blob_raises_legacy(self):
        """A short pickle blob (12 bytes) also raises LegacyEmbeddingError."""
        with pytest.raises(LegacyEmbeddingError):
            unpack_embedding(self._PICKLE_SINGLE_FLOAT)

    def test_legacy_error_is_value_error_subclass(self):
        """LegacyEmbeddingError is a ValueError so existing broad-except
        handlers still catch it, but callers can also match it
        specifically."""
        assert issubclass(LegacyEmbeddingError, ValueError)

    # ── False positives ────────────────────────────────────────────────
    #
    # A raw float32 vector is arbitrary binary.  Roughly one embedding in
    # 256 begins with the byte 0x80 by pure chance, and the sniff used to
    # test that byte alone — so those rows were refused as "legacy" and the
    # user was told to re-enrol a speaker whose data was perfectly intact.
    # CI caught it on a random 512-dim vector; these tests pin the shape of
    # the collision so it cannot come back.

    def test_float32_vector_starting_with_0x80_round_trips(self):
        """A valid embedding whose first byte is 0x80 must not be refused."""
        # b"\x80\x19\x97\xbf" is the exact little-endian float32 that failed
        # in CI.  The second element additionally collides with the pickle
        # protocol byte (0x02); only the missing trailing STOP separates
        # this blob from a protocol-2 pickle.
        raw = b"\x80\x19\x97\xbf\x80\x02\x00\x3f\x00\x00\x80\x3f"
        restored = unpack_embedding(raw, dimension=3)
        assert restored.dtype == np.float32
        assert restored.tobytes() == raw

    def test_0x80_prefix_without_pickle_protocol_byte_is_not_legacy(self):
        """0x80 followed by a byte no pickle protocol uses is float32 data."""
        # 0x19 is not in {2, 3, 4, 5}, so the protocol-byte check rejects it
        # even though the blob also happens to end in the STOP opcode 0x2e.
        raw = b"\x80\x19\x97\xbf\x00\x00\x00\x2e"
        assert unpack_embedding(raw, dimension=2).tobytes() == raw

    def test_pickle_prefix_without_stop_opcode_is_not_legacy(self):
        """A pickle frame header with no trailing STOP is not a pickle."""
        raw = b"\x80\x02\x97\xbf\x00\x00\x80\x3f"
        assert unpack_embedding(raw, dimension=2).tobytes() == raw

    def test_random_vectors_are_never_misread_as_legacy(self):
        """The whole point: no random embedding is refused as legacy.

        1024 draws would have produced ~4 spurious failures under the
        prefix-only check.
        """
        rng = np.random.default_rng(seed=20260824)
        for _ in range(1024):
            original = rng.standard_normal(512).astype(np.float32)
            restored = unpack_embedding(pack_embedding(original), dimension=512)
            np.testing.assert_array_equal(original, restored)


class TestAutoMatchResilience:
    """``_auto_match_speakers`` must skip one legacy speaker row without
    failing the whole job, and still match against the remaining valid
    known speakers."""

    @pytest.mark.asyncio
    async def test_skips_legacy_row_still_matches_valid(self, api):
        """One legacy known speaker + one valid known speaker → the valid
        one is still matched."""
        from artifice_transcribe.api.v1.routes import _auto_match_speakers
        from artifice_transcribe.db.models import (
            KnownSpeaker,
            SpeakerEmbedding,
            TranscriptionJob,
        )

        job_id = "job-resilience"
        # Matching float32 vectors: cosine sim ≈ 1.0 > THRESHOLD (0.65)
        emb_vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        emb_blob = pack_embedding(emb_vec)

        # Legacy pickle blob (starts with \x80)
        legacy_blob = TestLegacyPickleDetection._PICKLE_SINGLE_FLOAT

        async with api.session_factory() as db:
            # Create a completed job
            db.add(
                TranscriptionJob(
                    id=job_id,
                    filename="test.wav",
                    status=JobStatus.completed,
                )
            )
            await db.flush()

            # Add a job embedding and mapping for the auto-match
            db.add(
                SpeakerEmbedding(
                    job_id=job_id,
                    speaker_label="SPEAKER_00",
                    embedding=emb_blob,
                    dimension=4,
                )
            )
            db.add(
                SpeakerMapping(
                    job_id=job_id,
                    speaker_label="SPEAKER_00",
                    custom_name="SPEAKER_00",
                )
            )

            # Legacy known speaker (pickle bytes, will be skipped)
            db.add(
                KnownSpeaker(
                    id="ks-legacy",
                    name="Alice (legacy)",
                    embedding=legacy_blob,
                    model_name="pyannote/embedding",
                    dimension=512,
                )
            )

            # Valid known speaker (matching float32 bytes)
            db.add(
                KnownSpeaker(
                    id="ks-valid",
                    name="Bob",
                    embedding=emb_blob,
                    model_name="pyannote/embedding",
                    dimension=4,
                )
            )

            await db.commit()

        # Run auto-match — must not raise
        async with api.session_factory() as db:
            await _auto_match_speakers(job_id, db)

            # Verify Bob was matched (cosine sim = 1.0 > 0.65)
            mapping = (
                (
                    await db.execute(
                        select(SpeakerMapping).where(
                            SpeakerMapping.job_id == job_id,
                            SpeakerMapping.speaker_label == "SPEAKER_00",
                        )
                    )
                )
                .scalars()
                .first()
            )
            assert mapping is not None
            assert mapping.custom_name == "Bob"

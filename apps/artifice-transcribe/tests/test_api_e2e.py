# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end API tests — the automated counterpart of ``test_api.py``.

``test_api.py`` is a standalone script that verifies this same path against
a live server on 127.0.0.1:8000 (and, given an audio file, against the real
WhisperX engine). Everything it checks that does not require real model
weights is covered here through the ASGI app, with the transcription engine
faked at the same boundary the rest of the suite uses
(``routes._get_engine``).

No polling loop is needed, unlike the live-server script: Starlette runs
``BackgroundTasks`` before the ASGI app returns, and httpx's ASGITransport
awaits the app to completion, so the worker has already finished when the
POST response arrives.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytestmark = pytest.mark.asyncio


@dataclass
class FakeSegment:
    speaker: str
    start: float
    end: float
    text: str


@dataclass
class FakeResult:
    segments: list[FakeSegment]
    speaker_embeddings: dict[str, list[float]]


@dataclass
class FakeEngine:
    """Duck-type stand-in for the ASR backend (WhisperXEngine).

    Exposes exactly the surface the background worker touches:
    ``transcribe(**kwargs) -> result with .segments / .speaker_embeddings``
    and ``unload()``, which the worker must call in its ``finally`` block.
    """

    calls: list[str] = field(default_factory=list)
    unloaded: bool = False

    def transcribe(
        self,
        audio_path: str,
        *,
        language=None,
        min_speakers=None,
        max_speakers=None,
        progress_callback=None,
        custom_vocabulary=None,
        hotwords=None,
    ) -> FakeResult:
        self.calls.append(audio_path)
        return FakeResult(
            segments=[
                FakeSegment("SPEAKER_00", 0.0, 1.5, "Hello, this is a test recording."),
                FakeSegment("SPEAKER_01", 1.5, 3.25, "And this is the reply."),
            ],
            speaker_embeddings={"SPEAKER_00": [0.1] * 512},
        )

    def unload(self) -> None:
        self.unloaded = True


async def test_health(api):
    resp = await api.client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_missing_job_returns_404(api):
    resp = await api.client.get("/api/v1/jobs/nonexistent")
    assert resp.status_code == 404


async def test_transcribe_full_lifecycle(api, monkeypatch):
    """The script's full path: transcribe -> poll -> transcript -> rename
    speaker -> export (json/srt/vtt/txt) -> delete — without a live engine."""
    engine = FakeEngine()
    monkeypatch.setattr("artifice_transcribe.api.v1.routes._get_engine", lambda: engine)

    # 1. POST /api/v1/transcribe -> 202 + job_id. The background worker runs
    #    to completion inside this request cycle (see module docstring).
    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"fake-audio-data")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    # 2. Poll equivalent: the job completed with progress 100 and no error.
    status = (await api.client.get(f"/api/v1/jobs/{job_id}")).json()
    assert status["status"] == "completed", f"job ended as: {status}"
    assert status["progress_percentage"] == 100.0
    assert status["error_message"] is None
    # The worker was pointed at the uploaded file and unloaded the engine
    # afterwards (the VRAM-management contract in its `finally` block).
    assert engine.calls == [str(api.upload_dir / f"{job_id}_interview.wav")]
    assert engine.unloaded

    # 3. Transcript holds both segments; unmapped speaker labels pass through.
    transcript = (await api.client.get(f"/api/v1/jobs/{job_id}/transcript")).json()
    assert [s["text"] for s in transcript["segments"]] == [
        "Hello, this is a test recording.",
        "And this is the reply.",
    ]
    assert [s["speaker_label"] for s in transcript["segments"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]

    # 3b. Speaker embeddings returned by the engine were persisted.
    embeddings = (await api.client.get(f"/api/v1/jobs/{job_id}/speaker-embeddings")).json()
    assert [(e["speaker_label"], e["dimension"]) for e in embeddings] == [("SPEAKER_00", 512)]

    # 4. Rename a speaker; the transcript and exports use the custom name.
    renamed = await api.client.patch(
        f"/api/v1/jobs/{job_id}/speakers",
        json={"speakers": [{"speaker_label": "SPEAKER_00", "custom_name": "Test Speaker"}]},
    )
    assert renamed.status_code == 200
    transcript = (await api.client.get(f"/api/v1/jobs/{job_id}/transcript")).json()
    assert transcript["segments"][0]["speaker_label"] == "Test Speaker"

    # 5. The four export formats the script checks, in order.
    checks = {
        "json": ("application/json", '"speaker": "Test Speaker"'),
        "srt": ("text/srt", "00:00:00,000 --> 00:00:01,500"),
        "vtt": ("text/vtt", "WEBVTT"),
        "txt": ("text/plain", "[Test Speaker]"),
    }
    for fmt, (media_type, marker) in checks.items():
        resp = await api.client.get(f"/api/v1/jobs/{job_id}/export", params={"format": fmt})
        assert resp.status_code == 200, fmt
        assert resp.headers["content-type"].startswith(media_type), fmt
        assert marker in resp.text, fmt

    # 6. DELETE removes the job and its uploaded audio file.
    audio_file = api.upload_dir / f"{job_id}_interview.wav"
    assert audio_file.exists()
    resp = await api.client.delete(f"/api/v1/jobs/{job_id}")
    assert resp.status_code == 204
    assert (await api.client.get(f"/api/v1/jobs/{job_id}")).status_code == 404
    assert not audio_file.exists()


async def test_transcribe_marks_job_failed_when_engine_raises(api, monkeypatch):
    """The worker must not let an engine crash escape the background task —
    the job is marked failed with the error message instead."""

    class BoomEngine(FakeEngine):
        def transcribe(self, audio_path: str, **kwargs) -> FakeResult:
            raise RuntimeError("no model weights here")

    engine = BoomEngine()
    monkeypatch.setattr("artifice_transcribe.api.v1.routes._get_engine", lambda: engine)

    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"fake-audio-data")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]

    status = (await api.client.get(f"/api/v1/jobs/{job_id}")).json()
    assert status["status"] == "failed"
    assert "no model weights here" in status["error_message"]
    # unload() runs even on the failure path.
    assert engine.unloaded


async def test_transcribe_manual_mode_skips_engine(api, monkeypatch):
    """mode=manual creates a completed job with one empty segment and never
    touches the transcription engine or the background worker."""
    ran = []
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes._run_transcription",
        lambda *a, **k: ran.append(True),
    )

    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"fake-audio-data")},
        params={"mode": "manual"},
    )
    assert resp.status_code == 202
    assert resp.json()["status"] == "completed"
    job_id = resp.json()["job_id"]

    # No background task was queued, so the worker never ran.
    assert ran == []

    status = (await api.client.get(f"/api/v1/jobs/{job_id}")).json()
    assert status["status"] == "completed"
    assert status["progress_percentage"] == 100.0
    assert status["completed_at"] is not None
    assert status["error_message"] is None

    # Exactly one empty segment for the editor to open.
    transcript = (await api.client.get(f"/api/v1/jobs/{job_id}/transcript")).json()
    assert len(transcript["segments"]) == 1
    seg = transcript["segments"][0]
    assert seg["speaker_label"] == "SPEAKER_00"
    assert seg["start_time"] == 0.0
    assert seg["end_time"] == 0.0
    assert seg["text"] == ""

    # The uploaded audio is still stored for the editor's audio player.
    assert (api.upload_dir / f"{job_id}_interview.wav").exists()


async def test_transcribe_rejects_unknown_mode(api):
    """Any mode other than auto/manual is rejected with 400 before any work."""
    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"fake-audio-data")},
        params={"mode": "not-a-mode"},
    )
    assert resp.status_code == 400
    assert "mode" in resp.json()["detail"]

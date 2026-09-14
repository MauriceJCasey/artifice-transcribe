# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from pytest_httpx import HTTPXMock

from artifice_transcribe.db.models import JobStatus, SpeakerMapping, TranscriptionJob

pytestmark = pytest.mark.asyncio

_ASR_AVAILABLE = importlib.util.find_spec("torch") is not None


async def _make_job(
    api, job_id: str, filename: str, *, status=JobStatus.completed, created_at=None
):
    async with api.session_factory() as db:
        db.add(
            TranscriptionJob(
                id=job_id,
                filename=filename,
                status=status,
                progress_percentage=100.0 if status == JobStatus.completed else 0.0,
                created_at=created_at or datetime.now(timezone.utc),
            )
        )
        await db.commit()


# --------------------------------------------------------------- GET /jobs


async def test_list_jobs_empty(api):
    resp = await api.client.get("/api/v1/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_jobs_newest_first(api):
    now = datetime.now(timezone.utc)
    await _make_job(api, "job-older", "older.wav", created_at=now - timedelta(minutes=5))
    await _make_job(api, "job-newer", "newer.wav", created_at=now)

    resp = await api.client.get("/api/v1/jobs")
    assert resp.status_code == 200
    ids = [j["id"] for j in resp.json()]
    assert ids == ["job-newer", "job-older"]


# ------------------------------------------------------------ GET .../audio


async def test_get_audio_streams_file(api):
    await _make_job(api, "job-1", "clip.wav")
    audio_path = api.upload_dir / "job-1_clip.wav"
    audio_path.write_bytes(b"RIFF....WAVEfmt ")

    resp = await api.client.get("/api/v1/jobs/job-1/audio")
    assert resp.status_code == 200
    assert resp.content == b"RIFF....WAVEfmt "


async def test_get_audio_404_missing_job(api):
    resp = await api.client.get("/api/v1/jobs/nonexistent/audio")
    assert resp.status_code == 404


async def test_get_audio_404_missing_file_on_disk(api):
    await _make_job(api, "job-no-file", "ghost.wav")
    resp = await api.client.get("/api/v1/jobs/job-no-file/audio")
    assert resp.status_code == 404


# ---------------------------------------------------------- GET .../speakers


async def test_get_speakers_empty(api):
    await _make_job(api, "job-2", "clip2.wav")
    resp = await api.client.get("/api/v1/jobs/job-2/speakers")
    assert resp.status_code == 200
    assert resp.json() == {"job_id": "job-2", "speakers": []}


async def test_get_speakers_reflects_mappings(api):
    await _make_job(api, "job-3", "clip3.wav")
    async with api.session_factory() as db:
        db.add(SpeakerMapping(job_id="job-3", speaker_label="SPEAKER_00", custom_name="Alice"))
        await db.commit()

    resp = await api.client.get("/api/v1/jobs/job-3/speakers")
    assert resp.status_code == 200
    assert resp.json()["speakers"] == [{"speaker_label": "SPEAKER_00", "custom_name": "Alice"}]


async def test_get_speakers_reflects_patch(api):
    await _make_job(api, "job-4", "clip4.wav")
    await api.client.patch(
        "/api/v1/jobs/job-4/speakers",
        json={"speakers": [{"speaker_label": "SPEAKER_00", "custom_name": "Bob"}]},
    )

    resp = await api.client.get("/api/v1/jobs/job-4/speakers")
    assert resp.json()["speakers"] == [{"speaker_label": "SPEAKER_00", "custom_name": "Bob"}]


async def test_get_speakers_404_missing_job(api):
    resp = await api.client.get("/api/v1/jobs/nonexistent/speakers")
    assert resp.status_code == 404


# ── Helpers ──────────────────────────────────────────────────────────────────


def _leaked_outside(upload_dir: Path) -> bool:
    """True if any file was written outside the upload directory."""
    upload = upload_dir.resolve()
    for entry in upload.parent.rglob("*"):
        if entry.is_file() and upload not in entry.resolve().parents:
            # Ignore the test database created by the conftest fixture.
            if entry.suffix == ".db":
                continue
            return True
    return False


# ── Path traversal defences ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "malicious_filename,expected_status,description",
    [
        ("../../etc/passwd", 202, "dot-dot-slash is stripped to safe name"),
        ("..", 400, "bare dot-dot is rejected outright"),
        ("/etc/passwd", 202, "absolute path is stripped to safe name"),
        ("..\\..\\windows\\system32", 202, "backslash traversal is stripped to safe name"),
        ("", 422, "empty filename is rejected by FastAPI validation"),
    ],
)
async def test_transcribe_traversal_sanitisation(
    api, malicious_filename, expected_status, description
):
    """Malicious filenames are either rejected or safely stripped to their
    base name, and the written file stays inside the upload directory."""
    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": (malicious_filename, b"fake-audio-data")},
    )
    assert resp.status_code == expected_status, (
        f"Expected {expected_status} for {description}, got {resp.status_code}"
    )
    assert not _leaked_outside(api.upload_dir), (
        f"No files should exist outside upload_dir after {description}"
    )


@pytest.mark.parametrize(
    "name,fname,expected_status,description",
    [
        # Traversal via name is stripped to safe component -> accepted
        ("../../etc", "harmless.wav", 200, "traversal name stripped, file is safe"),
        # Traversal via filename with benign name -> accepted
        ("alice", "../../etc/passwd", 200, "traversal filename stripped, file is safe"),
        # Both fields bare dot-dot -> rejected
        ("..", "..", 400, "both fields are bare dot-dot -> rejected"),
        # Both fields empty -> rejected by FastAPI validation
        ("", "", 422, "both fields empty -> rejected"),
    ],
)
@pytest.mark.skipif(
    not _ASR_AVAILABLE, reason="ASR stack not installed (pyannote.audio unavailable)"
)
async def test_enroll_traversal_sanitisation(api, name, fname, expected_status, description):
    # Prevent the diarization model download (needs HF token) from
    # interfering with the sanitisation test.
    from unittest.mock import MagicMock, patch

    fake_embedding = MagicMock()
    fake_inference_cls = MagicMock(return_value=MagicMock(return_value=fake_embedding))
    fake_engine = MagicMock()
    fake_engine._diarize_model.model._embedding = MagicMock()

    with (
        patch("pyannote.audio.Inference", fake_inference_cls),
        patch(
            "artifice_transcribe.api.v1.routes._get_engine",
            return_value=fake_engine,
        ),
        patch(
            "artifice_transcribe.api.v1.routes.pack_embedding",
            return_value=b"\x00" * 2048,
        ),
    ):
        resp = await api.client.post(
            "/api/v1/speakers/enroll",
            data={"name": name},
            files={"file": (fname, b"fake-audio")},
        )
    assert resp.status_code == expected_status, (
        f"Expected {expected_status} for {description}, got {resp.status_code}"
    )
    assert not _leaked_outside(api.upload_dir), (
        f"No files should exist outside upload_dir after {description}"
    )


async def test_transcribe_sanitised_filename_written_inside_upload_dir(api):
    """A harmless filename must still round-trip correctly."""
    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"fake-audio-data")},
    )
    assert resp.status_code == 202
    job_id = resp.json()["job_id"]
    written = api.upload_dir / f"{job_id}_interview.wav"
    assert written.exists()
    assert written.read_bytes() == b"fake-audio-data"


# ── Size limit ───────────────────────────────────────────────────────────────


async def test_enroll_rejects_oversized_upload(api, monkeypatch):
    """The enrollment endpoint must enforce the same max_upload_size as /transcribe."""
    monkeypatch.setattr("artifice_transcribe.api.v1.routes.settings.max_upload_size", 10)
    resp = await api.client.post(
        "/api/v1/speakers/enroll",
        data={"name": "speaker"},
        files={"file": ("voice.wav", b"x" * 11)},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


async def test_transcribe_rejects_oversized_upload(api, monkeypatch):
    """The /transcribe endpoint must reject uploads larger than max_upload_size."""
    monkeypatch.setattr("artifice_transcribe.api.v1.routes.settings.max_upload_size", 10)
    resp = await api.client.post(
        "/api/v1/transcribe",
        files={"file": ("interview.wav", b"x" * 11)},
    )
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


# -------------------------------------------------------- streaming cap (bypass)


def test_read_capped_raises_during_read():
    """``read_capped`` raises ``UploadTooLarge`` once the limit is exceeded
    mid-stream, before the full body is gathered.

    This is a unit test on the shared helper; the endpoint-level tests above
    (``test_transcribe_rejects_oversized_upload`` and
    ``test_enroll_rejects_oversized_upload``) prove the routes translate it
    into HTTP 413.
    """
    from shared_ui.uploads import UploadTooLarge, read_capped

    class _FakeUpload:
        filename = "test.wav"

        def __init__(self, total: int):
            self._remain = total

        async def read(self, size: int = -1) -> bytes:
            if self._remain <= 0:
                return b""
            n = min(size if size > 0 else 4096, 4096, self._remain)
            self._remain -= n
            return b"x" * n

    with pytest.raises(UploadTooLarge):
        asyncio.run(read_capped(_FakeUpload(10_000), 10))


def test_read_capped_respects_dynamic_limit():
    """``read_capped`` uses the caller-supplied limit, not a hardcoded
    constant — important because transcribe's limit is configurable
    via settings.max_upload_size."""
    from shared_ui.uploads import UploadTooLarge, read_capped

    class _FakeUpload:
        filename = "test.wav"

        def __init__(self, total: int):
            self._remain = total

        async def read(self, size: int = -1) -> bytes:
            if self._remain <= 0:
                return b""
            n = min(size if size > 0 else 4096, 4096, self._remain)
            self._remain -= n
            return b"x" * n

    # Under a 1000-byte limit, 500 bytes should pass.
    result = asyncio.run(read_capped(_FakeUpload(500), 1000))
    assert len(result) == 500

    # At exactly the limit, should still pass.
    result2 = asyncio.run(read_capped(_FakeUpload(1000), 1000))
    assert len(result2) == 1000

    # One byte over must fail.
    with pytest.raises(UploadTooLarge):
        asyncio.run(read_capped(_FakeUpload(1001), 1000))


# --------------------------------------------------- inference endpoint shapes


async def test_inference_models_success_shape_matches_js(api, httpx_mock: HTTPXMock):
    """POST /api/v1/inference/models returns the keys app.js line 1565 reads."""
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "whisper-model"}, {"name": "llama3.2:3b"}]},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/v1/models",
        json={"data": [{"id": "openai-model"}]},
    )

    resp = await api.client.post(
        "/api/v1/inference/models",
        json={"base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Consumed by transcribe/web/static/js/app.js:1565
    assert set(data.keys()) == {"models"}
    assert sorted(data["models"]) == ["llama3.2:3b", "openai-model", "whisper-model"]


async def test_inference_models_failure_returns_empty_list(api, httpx_mock: HTTPXMock):
    """Unreachable server returns 500 so the frontend shows the error.

    probe_endpoint returns reachable=False as a normal ProbeResult, but
    get_available_models now raises RuntimeError when that happens.
    The route catches it and raises HTTP 500, matching the old behavior.
    """
    httpx_mock.add_exception(
        httpx.ConnectError("Connection refused"),
        url="http://localhost:11434/api/tags",
    )

    resp = await api.client.post(
        "/api/v1/inference/models",
        json={"base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
    )
    assert resp.status_code == 500
    data = resp.json()
    assert "detail" in data


async def test_inference_test_success_shape_matches_js(api, httpx_mock: HTTPXMock):
    """POST /api/v1/inference/test returns the keys app.js line 1596 reads."""
    httpx_mock.add_response(
        url="http://localhost:11434/api/tags",
        json={"models": [{"name": "model-a"}, {"name": "model-b"}]},
    )
    httpx_mock.add_response(
        url="http://localhost:11434/v1/models",
        json={"data": []},
    )

    resp = await api.client.post(
        "/api/v1/inference/test",
        json={"base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
    )
    assert resp.status_code == 200
    data = resp.json()
    # Consumed by transcribe/web/static/js/app.js:1596-1603
    assert set(data.keys()) == {"success", "message", "model_count"}
    assert data["success"] is True
    assert data["model_count"] == 2
    assert "Connected successfully" in data["message"]


async def test_inference_test_failure_shape_matches_js(api, httpx_mock: HTTPXMock):
    """Failure path keeps the same keys the JS consumes."""
    httpx_mock.add_exception(
        httpx.ConnectError("Connection refused"),
        url="http://localhost:11434/api/tags",
    )

    resp = await api.client.post(
        "/api/v1/inference/test",
        json={"base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) == {"success", "message", "model_count"}
    assert data["success"] is False
    assert data["model_count"] == 0
    assert "Server unreachable" in data["message"]


# -------------------------------------------------- CORS (ARTIFICE_CORS_ORIGINS)
#
# `main.app` builds its CORS middleware from the env var at *import* time, so
# exercising a non-default value means reloading the module after setting
# it. These tests use their own ASGITransport (bypassing the `api` fixture's
# DB wiring, which isn't needed here) and restore `main` to its unmodified
# state afterwards so other test modules that imported `app` at collection
# time are unaffected.


async def test_cors_default_origins_when_env_unset(monkeypatch):
    monkeypatch.delenv("ARTIFICE_CORS_ORIGINS", raising=False)
    import artifice_transcribe.main as main_module

    importlib.reload(main_module)
    try:
        transport = ASGITransport(app=main_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            allowed = await ac.get("/health", headers={"Origin": "http://localhost:8000"})
            assert allowed.headers.get("access-control-allow-origin") == "http://localhost:8000"

            rejected = await ac.get("/health", headers={"Origin": "http://evil.example"})
            assert "access-control-allow-origin" not in rejected.headers
    finally:
        importlib.reload(main_module)


async def test_cors_honors_env_override(monkeypatch):
    monkeypatch.setenv(
        "ARTIFICE_CORS_ORIGINS",
        "http://example.com:9999, http://foo.test:1234",
    )
    import artifice_transcribe.main as main_module

    importlib.reload(main_module)
    try:
        transport = ASGITransport(app=main_module.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            allowed = await ac.get("/health", headers={"Origin": "http://example.com:9999"})
            assert allowed.headers.get("access-control-allow-origin") == "http://example.com:9999"

            # The old hardcoded default must no longer be allowed once an
            # override is set — this is not additive.
            rejected = await ac.get("/health", headers={"Origin": "http://localhost:8000"})
            assert "access-control-allow-origin" not in rejected.headers
    finally:
        monkeypatch.undo()
        importlib.reload(main_module)

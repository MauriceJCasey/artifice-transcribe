# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Download-service API endpoint tests."""

import pytest
from artifice_transcribe.main import app
from httpx import ASGITransport, AsyncClient

# -- API endpoints -----------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models_returns_all_keys():
    """GET /api/v1/models returns all ASR model keys."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    keys = {m["key"] for m in data["models"]}
    assert "whisper-large-v3" in keys
    assert "pyannote-speaker-diarization" in keys
    assert "pyannote-embedding" in keys
    assert "parakeet-tdt-1.1b" in keys


@pytest.mark.asyncio
async def test_model_info_returns_transitive_sizes():
    """GET /api/v1/models/{key} includes dependencies and total size."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/pyannote-speaker-diarization")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) == 2
    assert data["total_size_bytes"] == 5_905_440 + 96_383_626
    assert data["requires_hf_token"] is True
    assert "cache_directory" in data


@pytest.mark.asyncio
async def test_model_info_unknown_key_returns_404():
    """GET /api/v1/models/unknown returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/nonexistent")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_consent_grant_and_revoke(clean_consent):
    """POST /api/v1/models/{key}/consent grants and revokes."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/whisper-large-v3/consent",
            json={"consent": True},
        )
    assert resp.status_code == 200
    assert resp.json()["consented"] is True

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3")
    assert resp.json()["consented"] is True

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/whisper-large-v3/consent",
            json={"consent": False},
        )
    assert resp.status_code == 200
    assert resp.json()["consented"] is False


@pytest.mark.asyncio
async def test_download_refused_without_consent():
    """POST /api/v1/models/{key}/download returns 403 without consent."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/models/whisper-large-v3/download")
    assert resp.status_code == 403
    assert "Consent has not been granted" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_download_refused_for_unknown_key():
    """POST /api/v1/models/nonexistent/download returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/models/nonexistent/download")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_download_status_never_started():
    """GET /api/v1/models/{key}/download/status returns never_started."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/whisper-large-v3/download/status")
    assert resp.status_code == 200
    assert resp.json()["status"] == "never_started"


@pytest.mark.asyncio
async def test_consent_endpoint_unknown_key():
    """POST consent for unknown key returns 404."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/models/nonexistent/consent",
            json={"consent": True},
        )
    assert resp.status_code == 404

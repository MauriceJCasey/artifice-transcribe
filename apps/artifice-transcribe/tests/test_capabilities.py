# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the ASR-disabled feature: capabilities endpoint and the
AsrUnavailable error-handling paths across the API.

No test in this file imports the actual ASR stack (torch / whisperx /
pyannote.audio).  The disabled state is the default experience for a
base install — it is a shipped feature, not an error path — and these
tests prove it stays intact across refactors.
"""

from __future__ import annotations

import importlib.util

import pytest
from artifice_transcribe.api.v1.routes import AsrUnavailable

# ── GET /api/v1/capabilities ──────────────────────────────────────────────


@pytest.mark.asyncio
class TestCapabilitiesGet:
    """GET /api/v1/capabilities reports whether the ASR stack is installed."""

    async def test_asr_unavailable_when_packages_are_missing(self, api, monkeypatch):
        """When none of the ASR packages can be found, capabilities reports
        unavailable with a reason and install hint."""
        original = importlib.util.find_spec

        def _missing_find_spec(name, package=None):
            if name in ("whisperx", "torch", "torchaudio", "torchvision", "torchcodec"):
                return None
            return original(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", _missing_find_spec)

        resp = await api.client.get("/api/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "asr" in data
        assert data["asr"]["available"] is False
        assert "reason" in data["asr"], "missing reason for asr unavailability"
        assert isinstance(data["asr"]["reason"], str)
        assert len(data["asr"]["reason"]) > 0
        assert "install_hint" in data["asr"]
        assert "uv sync" in data["asr"]["install_hint"]

    async def test_asr_available_when_all_packages_are_found(self, api, monkeypatch):
        """When every required package is importable, capabilities reports
        available with no reason or install hint keys."""
        original = importlib.util.find_spec

        def _found_find_spec(name, package=None):
            if name in ("whisperx", "torch", "torchaudio", "torchvision", "torchcodec"):
                return original("sys", None)  # any non-None value will do
            return original(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", _found_find_spec)

        resp = await api.client.get("/api/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert "asr" in data
        assert data["asr"]["available"] is True, f"expected available=True, got {data['asr']}"
        # No reason or install_hint when the stack is available.
        assert "reason" not in data["asr"]
        assert "install_hint" not in data["asr"]

    async def test_partial_install_reports_unavailable(self, api, monkeypatch):
        """If torch is present but whisperx (or torchcodec) is missing,
        capabilities correctly reports unavailable — a partial install is
        not a working install."""
        original = importlib.util.find_spec

        def _partial_find_spec(name, package=None):
            if name in ("torch", "torchaudio", "torchvision"):
                return original("sys", None)
            if name in ("whisperx", "torchcodec"):
                return None
            return original(name, package)

        monkeypatch.setattr(importlib.util, "find_spec", _partial_find_spec)

        resp = await api.client.get("/api/v1/capabilities")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asr"]["available"] is False, "partial install must report unavailable"


# ── AsrUnavailable error handling across engine-dependent routes ──────────


@pytest.mark.asyncio
class TestAsrUnavailableHandlers:
    """Routes that depend on the transcription engine must return a clear,
    structured response (not a 500 traceback) when the ASR stack is missing."""

    async def test_health_detailed_degraded_when_asr_unavailable(self, api, monkeypatch):
        """GET /health/detailed must return 'degraded' with engine unavailable
        and a correct database status when the ASR stack cannot be loaded."""
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._get_engine",
            _raise_asr_unavailable,
        )

        resp = await api.client.get("/api/v1/health/detailed")
        assert resp.status_code == 200
        data = resp.json()

        assert data["status"] == "degraded", f"expected 'degraded', got {data['status']!r}"
        assert data["engine"]["available"] is False
        assert "install_hint" in data["engine"]
        assert "uv sync" in data["engine"]["install_hint"]
        # The test database (isolated by conftest) should still be healthy.
        assert data["database"]["status"] == "ok", (
            "test database should report ok independently of ASR stack"
        )

    async def test_health_preload_returns_error_when_asr_unavailable(self, api, monkeypatch):
        """POST /health/preload catches AsrUnavailable and returns a 200 with
        ok=False and a clear error message — it must not surface a traceback."""
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._get_engine",
            _raise_asr_unavailable,
        )

        resp = await api.client.post("/api/v1/health/preload")
        assert resp.status_code == 200
        data = resp.json()

        assert data["ok"] is False
        assert "error" in data
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0
        assert "install_hint" in data
        assert "uv sync" in data["install_hint"]

    async def test_config_update_rejects_with_503_when_asr_unavailable(self, api, monkeypatch):
        """PATCH /api/v1/config triggers an engine reload when the whisper
        model changes.  An AsrUnavailable during that reload must return
        HTTP 503, not 500."""

        def _fail_on_reload():
            raise AsrUnavailable()

        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._reload_engine",
            _fail_on_reload,
        )

        resp = await api.client.patch(
            "/api/v1/config",
            json={"whisper_model": "large-v3"},
        )
        assert resp.status_code == 503, (
            f"expected 503 for ASR-unavailable config update, got {resp.status_code}"
        )
        detail = resp.json().get("detail", "")
        assert "not installed" in detail.lower() or ("uv sync" in detail.lower()), (
            f"unexpected error detail: {detail!r}"
        )

    async def test_enroll_speaker_returns_503_when_asr_unavailable(self, api, monkeypatch):
        """POST /api/v1/speakers/enroll must return HTTP 503 when the engine
        cannot be loaded, rather than surfacing a raw AsrUnavailable traceback."""
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._get_engine",
            _raise_asr_unavailable,
        )

        resp = await api.client.post(
            "/api/v1/speakers/enroll",
            data={"name": "test-speaker"},
            files={"file": ("voice.wav", b"fake-audio-data")},
        )
        assert resp.status_code == 503, (
            f"expected 503 for ASR-unavailable enroll, got {resp.status_code}"
        )
        detail = resp.json().get("detail", "")
        assert "not installed" in detail.lower() or ("uv sync" in detail.lower()), (
            f"unexpected error detail: {detail!r}"
        )


# ── Helpers ───────────────────────────────────────────────────────────────


def _raise_asr_unavailable():
    """Helper to use as a monkeypatched _get_engine target."""
    raise AsrUnavailable()


# ── asr_backend config wiring (Parakeet engine selection) ─────────────────


@pytest.mark.asyncio
class TestAsrBackendConfig:
    """GET/PATCH /api/v1/config must expose and honour ``asr_backend`` the same
    way it does ``whisper_model`` — without importing the ASR stack."""

    async def test_get_config_includes_asr_backend(self, api):
        resp = await api.client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asr_backend"] == "whisperx"  # default preserves behavior
        assert set(data["available_asr_backends"]) == {"whisperx", "parakeet"}

    async def test_patch_asr_backend_triggers_reload(self, api, monkeypatch):
        from artifice_transcribe.config import settings

        # Snapshot/restore the singleton so this test cannot leak a backend
        # change into later tests.
        monkeypatch.setattr(settings, "asr_backend", settings.asr_backend)

        reloaded = []

        async def _reload():
            reloaded.append(True)

        monkeypatch.setattr("artifice_transcribe.api.v1.routes._reload_engine", _reload)

        resp = await api.client.patch("/api/v1/config", json={"asr_backend": "parakeet"})
        assert resp.status_code == 200
        assert reloaded == [True], "changing asr_backend must reload the engine"

    async def test_patch_rejects_unknown_asr_backend(self, api):
        resp = await api.client.patch("/api/v1/config", json={"asr_backend": "not-a-backend"})
        assert resp.status_code == 400
        assert "not-a-backend" in resp.json()["detail"]

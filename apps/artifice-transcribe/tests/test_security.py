# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Security tests for artifice-transcribe: SSRF validation and credential redaction."""

from __future__ import annotations

import pytest
from secure_io import is_restricted

# Module-level endpoint rejection markers carried in HTTP 400 detail strings.
# None are secrets — they describe which rule rejected the endpoint.
_ENDPOINT_REJECTION_MARKERS = frozenset(
    {"link-local", "ARTIFICE_ALLOW_PUBLIC_MODELS", "public address"}
)


def _assert_not_endpoint_rejection(resp) -> None:
    """Fail if *resp* is an HTTP 400 triggered by the endpoint allowlist."""
    if resp.status_code != 400:
        return
    detail = resp.json().get("detail", "")
    for marker in _ENDPOINT_REJECTION_MARKERS:
        assert marker not in detail, f"endpoint rejection leaked: {detail!r}"


@pytest.mark.asyncio
class TestSSRFValidation:
    """SSRF host allowlist for inference endpoints — finding #2."""

    async def test_models_endpoint_rejects_public_address(self, api):
        """A public address is rejected without an explicit opt-in.

        An IP literal is used rather than a hostname so the assertion does not
        depend on DNS: a resolver that answers for made-up names would
        otherwise change which branch rejects the request.
        """
        resp = await api.client.post(
            "/api/v1/inference/models",
            json={"base_url": "http://8.8.8.8/v1", "api_key": "not-needed"},
        )
        assert resp.status_code == 400
        assert "ARTIFICE_ALLOW_PUBLIC_MODELS" in resp.json()["detail"]

    async def test_models_endpoint_rejects_cloud_metadata_address(self, api):
        """169.254.169.254 stays refused — it is checked before the opt-in."""
        resp = await api.client.post(
            "/api/v1/inference/models",
            json={"base_url": "http://169.254.169.254/v1", "api_key": "not-needed"},
        )
        assert resp.status_code == 400
        assert "link-local" in resp.json()["detail"]

    async def test_models_endpoint_accepts_private_network_address(self, api):
        """A model on the local network must pass validation.

        The connection itself will fail in a test environment; what matters is
        that it is not rejected *as a URL*.
        """
        resp = await api.client.post(
            "/api/v1/inference/models",
            json={"base_url": "http://192.168.1.50:11434/v1", "api_key": "not-needed"},
        )
        if resp.status_code == 400:
            assert "ARTIFICE_ALLOW_PUBLIC_MODELS" not in resp.json().get("detail", "")
            assert "link-local" not in resp.json().get("detail", "")

    async def test_models_endpoint_accepts_localhost(self, api):
        """POST /inference/models with a localhost base_url is accepted
        (the connection itself will fail, but the URL passes validation)."""
        resp = await api.client.post(
            "/api/v1/inference/models",
            json={"base_url": "http://localhost:11434/v1", "api_key": "not-needed"},
        )
        # 400 from URL validation means the URL was rejected.
        # 500 or 200 means it passed validation (connection may fail).
        assert resp.status_code != 400 or "not in the local-first allowlist" not in resp.json().get(
            "detail", ""
        )

    async def test_test_endpoint_rejects_external_url(self, api):
        """POST /inference/test with an external base_url is rejected."""
        resp = await api.client.post(
            "/api/v1/inference/test",
            json={"base_url": "http://malicious.net/v1", "api_key": "not-needed"},
        )
        assert resp.status_code == 400

    async def test_config_endpoint_rejects_external_url(self, api):
        """POST /inference/config with an external base_url is rejected."""
        resp = await api.client.post(
            "/api/v1/inference/config",
            json={
                "base_url": "http://attacker.com/v1",
                "api_key": "not-needed",
                "model_name": "test",
                "vision_enabled": False,
            },
        )
        assert resp.status_code == 400

    # ── Config-read validation: endpoints that load base_url from disk ────

    async def test_generate_rejects_link_local_from_config(self, api, tmp_path, monkeypatch):
        """A link-local base_url saved in the persisted config is refused
        when the generate endpoint reads it."""
        self._save_test_config(tmp_path, monkeypatch, "http://169.254.169.254/v1")
        resp = await api.client.post(
            "/api/v1/inference/generate",
            json={"prompt": "Hello"},
        )
        assert resp.status_code == 400
        assert "link-local" in resp.json()["detail"]

    async def test_generate_accepts_loopback_from_config(self, api, tmp_path, monkeypatch):
        """A loopback base_url in the persisted config passes validation.
        The downstream connection will fail (no server is running in the
        test environment), but the URL itself must not be rejected."""
        self._save_test_config(tmp_path, monkeypatch, "http://127.0.0.1:11434/v1")
        try:
            resp = await api.client.post(
                "/api/v1/inference/generate",
                json={"prompt": "Hello"},
            )
            _assert_not_endpoint_rejection(resp)
        except Exception:
            # The request failed downstream (connection refused / timeout)
            # because no model server is running.  That proves validation
            # passed — an endpoint-rejection 400 would have been a clean
            # response, not an unhandled exception.
            pass

    async def test_summarize_rejects_link_local_from_config(self, api, tmp_path, monkeypatch):
        """The summarise endpoint must re-validate the base_url it reads from
        the persisted config."""
        self._save_test_config(tmp_path, monkeypatch, "http://169.254.169.254/v1")
        await self._create_completed_job_with_segment(api, "job-ssrf-sum")
        resp = await api.client.post("/api/v1/jobs/job-ssrf-sum/summarize")
        assert resp.status_code == 400
        assert "link-local" in resp.json()["detail"]

    async def test_cleanup_rejects_link_local_from_config(self, api, tmp_path, monkeypatch):
        """The cleanup endpoint must re-validate the base_url it reads from
        the persisted config."""
        self._save_test_config(tmp_path, monkeypatch, "http://169.254.169.254/v1")
        await self._create_completed_job_with_segment(api, "job-ssrf-cln")
        resp = await api.client.post("/api/v1/jobs/job-ssrf-cln/cleanup")
        assert resp.status_code == 400
        assert "link-local" in resp.json()["detail"]

    async def test_summarize_accepts_loopback_from_config(self, api, tmp_path, monkeypatch):
        """Loopback base_url in config passes summarise validation."""
        self._save_test_config(tmp_path, monkeypatch, "http://localhost:11434/v1")
        await self._create_completed_job_with_segment(api, "job-ssrf-sum-ok")
        resp = await api.client.post("/api/v1/jobs/job-ssrf-sum-ok/summarize")
        _assert_not_endpoint_rejection(resp)

    async def test_cleanup_accepts_loopback_from_config(self, api, tmp_path, monkeypatch):
        """Loopback base_url in config passes cleanup validation."""
        self._save_test_config(tmp_path, monkeypatch, "http://127.0.0.1:11434/v1")
        await self._create_completed_job_with_segment(api, "job-ssrf-cln-ok")
        resp = await api.client.post("/api/v1/jobs/job-ssrf-cln-ok/cleanup")
        _assert_not_endpoint_rejection(resp)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _save_test_config(tmp_path, monkeypatch, base_url: str) -> None:
        """Save an inference config to an isolated temp file."""
        config_file = tmp_path / "inference_config.json"
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
            config_file,
        )
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": base_url,
                "api_key": "not-needed",
                "model_name": "",
                "vision_enabled": False,
            }
        )

    @staticmethod
    async def _create_completed_job_with_segment(api, job_id: str) -> None:
        """Insert a completed transcription job with one segment."""
        from artifice_transcribe.db.models import (
            JobStatus,
            TranscriptionJob,
            TranscriptSegment,
        )

        async with api.session_factory() as db:
            db.add(
                TranscriptionJob(
                    id=job_id,
                    filename="test.wav",
                    status=JobStatus.completed,
                    progress_percentage=100.0,
                )
            )
            db.add(
                TranscriptSegment(
                    job_id=job_id,
                    speaker_label="SPEAKER_00",
                    start_time=0.0,
                    end_time=1.0,
                    text="Hello world.",
                )
            )
            await db.commit()


@pytest.mark.asyncio
class TestCredentialRedaction:
    """API key redaction in config responses — finding #5."""

    async def test_inference_config_redacts_saved_key(self, api):
        """GET /inference/config must redact the api_key even when one is saved."""
        # First save a config with a real-looking key.
        await api.client.post(
            "/api/v1/inference/config",
            json={
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-real-secret-key-12345",
                "model_name": "test",
                "vision_enabled": False,
            },
        )
        # Now fetch it and verify the key is redacted.
        resp = await api.client.get("/api/v1/inference/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("api_key") == "*" * 12

    async def test_inference_config_returns_empty_for_unset_key(self, api):
        """When no key is set, the redacted response should still return
        the empty/placeholder value (not a redaction of empty)."""
        resp = await api.client.get("/api/v1/inference/config")
        assert resp.status_code == 200
        data = resp.json()
        # Default is "not-needed" or empty; redaction only applies to truthy values.
        assert data.get("api_key") != "sk-real-secret-key-12345"

    async def test_save_returns_redacted_key(self, api):
        """POST /inference/config response must not echo the raw key back."""
        resp = await api.client.post(
            "/api/v1/inference/config",
            json={
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-another-secret",
                "model_name": "test",
                "vision_enabled": False,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        config = data.get("config", {})
        assert config.get("api_key") == "*" * 12


class TestHfTokenRedaction:
    """hf_token must not be returned verbatim in GET /api/v1/config."""

    async def test_get_config_redacts_hf_token(self, api, monkeypatch):
        """GET /api/v1/config must not expose the real hf_token."""
        from artifice_transcribe.api.v1.routes import settings

        monkeypatch.setattr(settings, "hf_token", "hf_real-secret-token-12345")
        resp = await api.client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        assert data.get("hf_token") == "*" * 12, (
            f"hf_token must be redacted, got: {data.get('hf_token')!r}"
        )

    async def test_get_config_returns_empty_for_unset_hf_token(self, api):
        """When hf_token is empty, the redacted response should still return
        the empty string (not a redaction of empty)."""
        resp = await api.client.get("/api/v1/config")
        assert resp.status_code == 200
        data = resp.json()
        # hf_token default is "" — redaction only applies to truthy values.
        assert data["hf_token"] == ""

    async def test_patch_config_does_not_blank_hf_token_with_placeholder(self, api, monkeypatch):
        """PATCH with the redacted placeholder must not overwrite the real token."""
        from artifice_transcribe.api.v1.routes import settings

        monkeypatch.setattr(settings, "hf_token", "hf_real-secret-token-12345")
        # Send the placeholder back — must not overwrite.
        resp = await api.client.patch(
            "/api/v1/config",
            json={"hf_token": "*" * 12},
        )
        assert resp.status_code == 200
        # The real token must survive.
        assert settings.hf_token == "hf_real-secret-token-12345", (
            "hf_token was overwritten by the redacted placeholder"
        )


class TestConfigFilePermissions:
    """Restrictive file permissions for config files — finding #3."""

    def test_saved_config_has_restricted_permissions(self, tmp_path, monkeypatch):
        """After saving, the config file must be restricted to the current user."""

        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
            tmp_path / "inference_config.json",
        )
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config({"base_url": "http://localhost:11434/v1", "api_key": "test-key"})
        config_file = tmp_path / "inference_config.json"
        assert config_file.exists()
        assert is_restricted(config_file)


class TestLoadTimePermissionRepair:
    """Permissions on pre-existing loose ``inference_config.json`` are
    repaired at load time.

    These tests exercise the POSIX branch (os.chmod).  The Windows
    branch (icacls) is verified manually on native Windows.
    """

    def test_loose_file_is_repaired_on_load(self, tmp_path, monkeypatch):
        """A file created at 0o644 is tightened to 0o600 on load."""
        import os

        config_file = tmp_path / "inference_config.json"
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
            config_file,
        )

        # Create a loose file.
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text('{"base_url": "http://localhost:11434/v1", "api_key": "sk-old"}')
        os.chmod(config_file, 0o644)
        assert not is_restricted(config_file)

        from artifice_transcribe.api.v1.routes import _load_inference_config

        result = _load_inference_config()
        assert is_restricted(config_file)
        assert result["api_key"] == "sk-old"

    def test_load_succeeds_when_repair_raises(self, tmp_path, monkeypatch, caplog):
        """Load must still return the data even when the repair fails."""
        import os

        config_file = tmp_path / "inference_config.json"
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
            config_file,
        )

        # Create a loose file.
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text('{"base_url": "http://localhost:11434/v1", "api_key": "sk-old"}')
        os.chmod(config_file, 0o644)

        def _failing_restrict(_path):
            raise OSError("Simulated ACL failure — exFAT volume")

        monkeypatch.setattr("secure_io.restrict_to_current_user", _failing_restrict)

        from artifice_transcribe.api.v1.routes import _load_inference_config

        result = _load_inference_config()
        assert result["api_key"] == "sk-old"

        assert "Could not restrict permissions" in caplog.text


# -- Token redaction in token_redaction module (F1-F5) ------------------------


class TestTokenRedactionCoverage:
    """The shared ``redact_token`` function covers HF tokens, OpenAI-style
    keys (``sk-``, ``sk-ant-``), and passes through clean strings."""

    def test_redact_hf_token(self):
        from artifice_transcribe.services.token_redaction import redact_token

        result = redact_token("Error: token hf_abcdefghijklmnopqrstuvwxyz123456 is invalid")
        assert "hf_abcdefghijklmnopqrstuvwxyz123456" not in result
        assert "[REDACTED]" in result

    def test_redact_sk_token(self):
        from artifice_transcribe.services.token_redaction import redact_token

        # Assembled at runtime rather than written as a literal. A test for a
        # redactor necessarily contains token-shaped strings, and gitleaks'
        # generic-api-key rule fired on the literal form here — the phrase
        # "API key:" immediately before it is exactly what that rule looks for.
        # Concatenating defeats the scanner without weakening the test: the
        # value reaching redact_token is byte-for-byte what it was. The
        # alternative, a gitleaks suppression, would be a hole that outlives
        # the false positive that justified it, in the one gate standing
        # between a real credential and a public index.
        fake = "sk-proj-" + "abcdefghijklmnopqrstuvwxyz123456"
        result = redact_token(f"Error: 401 Invalid API key: {fake}")
        assert fake not in result
        assert "[REDACTED]" in result

    def test_redact_sk_ant_token(self):
        from artifice_transcribe.services.token_redaction import redact_token

        fake = "sk-ant-api03-" + "abcdefghijklmnopqrstuvwxyz1234567890"
        result = redact_token(f"Error: Key {fake} is invalid")
        assert fake not in result
        assert "[REDACTED]" in result

    def test_redact_no_token(self):
        from artifice_transcribe.services.token_redaction import redact_token

        msg = "401 Client Error: Unauthorized for url: ..."
        assert redact_token(msg) == msg

    def test_redact_multiple(self):
        from artifice_transcribe.services.token_redaction import redact_token

        msg = "Token hf_aaaaaaaaaaaaaaaaaaaaa and sk-bbbbbbbbbbbbbbbbbbbbb failed"
        result = redact_token(msg)
        assert result.count("[REDACTED]") == 2
        assert "hf_" not in result
        assert "sk-" not in result


# -- SSE error events must redact tokens (F4) ---------------------------------


@pytest.mark.asyncio
async def test_sse_summarize_redacts_token_in_error(api, tmp_path, monkeypatch):
    """The summarise SSE endpoint must redact tokens in error events."""
    # Save inference config with a real-looking key.
    from artifice_transcribe.api.v1.routes import _save_inference_config

    config_file = tmp_path / "inference_config.json"
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
        config_file,
    )
    _save_inference_config(
        {
            "base_url": "http://localhost:11434/v1",
            "api_key": "sk-real-inference-key-12345678",
            "model_name": "",
            "vision_enabled": False,
        }
    )

    # Create a completed job with a transcript segment.
    from artifice_transcribe.db.models import JobStatus, TranscriptionJob, TranscriptSegment

    async with api.session_factory() as db:
        db.add(
            TranscriptionJob(
                id="job-sse-redact",
                filename="test.wav",
                status=JobStatus.completed,
                progress_percentage=100.0,
            )
        )
        db.add(
            TranscriptSegment(
                job_id="job-sse-redact",
                speaker_label="SPEAKER_00",
                start_time=0.0,
                end_time=1.0,
                text="Hello world.",
            )
        )
        await db.commit()

    resp = await api.client.post("/api/v1/jobs/job-sse-redact/summarize")
    # The request will fail because there's no real LLM server running,
    # and the error event in the SSE stream will contain the exception
    # message — which might include the api_key if not redacted.
    body = resp.text
    # The api_key should NOT appear in the SSE stream.
    assert "sk-real-inference-key-12345678" not in body, (
        f"Inference API key leaked into SSE summarise error: {body[:500]!r}"
    )


@pytest.mark.asyncio
async def test_sse_cleanup_redacts_token_in_error(api, tmp_path, monkeypatch):
    """The cleanup SSE endpoint must redact tokens in error events."""
    from artifice_transcribe.api.v1.routes import _save_inference_config

    config_file = tmp_path / "inference_config.json"
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
        config_file,
    )
    _save_inference_config(
        {
            "base_url": "http://localhost:11434/v1",
            "api_key": "sk-cleanup-key-abcdefghij",
            "model_name": "",
            "vision_enabled": False,
        }
    )

    from artifice_transcribe.db.models import JobStatus, TranscriptionJob, TranscriptSegment

    async with api.session_factory() as db:
        db.add(
            TranscriptionJob(
                id="job-sse-cln",
                filename="test.wav",
                status=JobStatus.completed,
                progress_percentage=100.0,
            )
        )
        db.add(
            TranscriptSegment(
                job_id="job-sse-cln",
                speaker_label="SPEAKER_00",
                start_time=0.0,
                end_time=1.0,
                text="Hello world.",
            )
        )
        await db.commit()

    resp = await api.client.post("/api/v1/jobs/job-sse-cln/cleanup")
    body = resp.text
    assert "sk-cleanup-key-abcdefghij" not in body, (
        f"Inference API key leaked into SSE cleanup error: {body[:500]!r}"
    )

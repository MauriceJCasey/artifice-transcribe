# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the security-hardening batch (PR4): three independent fixes.

1. A loopback-only bind guard for the native CLI (``main._assert_loopback_host``).
2. URL ``user:pass@`` credential stripping in ``test_connection`` error text.
3. Warning logging when the on-disk HF-token / inference-config files are unreadable.
"""

from __future__ import annotations

import pytest

# ── Fix 1: loopback-only bind guard ──────────────────────────────────────────


class TestLoopbackHostGuard:
    def test_loopback_hosts_are_accepted(self):
        from artifice_transcribe.main import _assert_loopback_host

        for host in ("127.0.0.1", "localhost", "::1", "[::1]"):
            _assert_loopback_host(host)  # must not raise

    def test_non_loopback_host_is_refused(self):
        from artifice_transcribe.main import _assert_loopback_host

        with pytest.raises(SystemExit):
            _assert_loopback_host("0.0.0.0")


# ── Fix 2: URL userinfo stripping ────────────────────────────────────────────


class TestStripUrlUserinfo:
    def test_removes_credentials_from_embedded_url(self):
        from artifice_transcribe.services.token_redaction import strip_url_userinfo

        text = "Connection failed for http://user:pass@evil:1234/v1"
        result = strip_url_userinfo(text)
        assert "user:pass" not in result
        assert "http://evil:1234/v1" in result

    def test_passthrough_when_no_url(self):
        from artifice_transcribe.services.token_redaction import strip_url_userinfo

        text = "No URL here"
        assert strip_url_userinfo(text) == text


class TestConnectionCredentialRedaction:
    async def test_test_connection_strips_url_userinfo_from_exception(self, monkeypatch):
        """``test_connection`` must not echo ``user:pass@`` back to the client.

        ``probe_endpoint`` is imported at call time (not module import), so the
        mock patches the real ``model_harness.discovery.probe_endpoint`` symbol.
        """
        from artifice_transcribe.services.inference import test_connection

        def _boom(base_url, policy=None, timeout_s=None):
            raise RuntimeError("Connection refused for http://user:pass@evil:1234/v1")

        monkeypatch.setattr("model_harness.discovery.probe_endpoint", _boom)

        result = await test_connection("http://localhost:11434/v1")

        assert result["success"] is False
        assert "user:pass" not in result["message"]
        assert "pass" not in result["message"]
        # The failure must still be communicated, not blanked out entirely.
        assert "Server unreachable" in result["message"]


# ── Fix 3: config-load failure visibility ────────────────────────────────────


class TestConfigLoadWarning:
    def test_load_hf_token_logs_warning_on_corrupt_file(self, tmp_path, monkeypatch, caplog):
        from artifice_transcribe.api.v1.routes import _load_hf_token, settings

        monkeypatch.setattr(settings, "hf_token", "")
        token_file = tmp_path / "hf_token.json"
        token_file.write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setattr("artifice_transcribe.api.v1.routes._HF_TOKEN_FILE", token_file)

        result = _load_hf_token()

        assert result == ""
        assert "Could not read" in caplog.text

    def test_load_inference_config_logs_warning_on_corrupt_file(
        self, tmp_path, monkeypatch, caplog
    ):
        from artifice_transcribe.api.v1.routes import _load_inference_config

        config_file = tmp_path / "inference_config.json"
        config_file.write_text("{ not valid json", encoding="utf-8")
        monkeypatch.setattr(
            "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
            config_file,
        )

        result = _load_inference_config()

        assert result == {
            "base_url": "http://localhost:11434/v1",
            "api_key": "not-needed",
            "model_name": "",
            "vision_enabled": False,
        }
        assert "Could not read" in caplog.text

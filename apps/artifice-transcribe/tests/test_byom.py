# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the BYOM onboarding endpoints in artifice-transcribe."""

from __future__ import annotations

import json
import time
from unittest.mock import patch

import httpx
import pytest
from artifice_transcribe.main import app
from fastapi.testclient import TestClient
from model_harness.discovery import ProbeResult


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Isolate inference config so no test touches the real file.

    ``byom.py`` imports ``_load_inference_config`` and
    ``_save_inference_config`` from ``routes.py``.  Both functions access
    ``routes._INFERENCE_CONFIG_FILE``, so patching that one module-level
    constant redirects all callers.
    """
    cfg_file = tmp_path / "inference_config.json"
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes._INFERENCE_CONFIG_FILE",
        cfg_file,
    )
    # Used as a context manager so the app's lifespan actually runs. The
    # lifespan (main.py:28-38) is what creates the database tables, and
    # ``TestClient(app)`` alone does not trigger it. Without this,
    # ``/api/v1/jobs`` raises from SQLAlchemy on any machine whose database
    # does not already exist — which is every CI runner, while passing on a
    # developer's machine that has one left over from an earlier run.
    with TestClient(app) as test_client:
        yield test_client


# ── GET /api/byom/state ───────────────────────────────────────────────────


class TestByomState:
    def test_returns_all_keys_in_default_state(self, client):
        r = client.get("/api/byom/state")
        assert r.status_code == 200
        body = r.json()

        assert body["app"] == "artifice-transcribe"
        assert body["configured"] is False
        # Default inference config has base_url set; with no file it falls
        # back to the built-in default.
        assert body["endpoint"] == "http://localhost:11434/v1"
        assert body["model"] is None
        assert "recommendations" in body

    def test_configured_true_when_api_key_set(self, client):
        """A real (non-default) api_key makes configured=True."""
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": "http://localhost:11434/v1",
                "api_key": "sk-real",
                "model_name": "",
                "vision_enabled": False,
            }
        )

        r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["configured"] is True

    def test_configured_false_for_default_api_key(self, client):
        """The default 'not-needed' api_key does not count as configured."""
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": "http://localhost:11434/v1",
                "api_key": "not-needed",
                "model_name": "",
                "vision_enabled": False,
            }
        )

        r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["configured"] is False

    def test_configured_true_when_base_url_changed(self, client):
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": "http://localhost:9999/v1",
                "api_key": "not-needed",
                "model_name": "",
                "vision_enabled": False,
            }
        )

        r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["configured"] is True

    def test_model_returned_from_config(self, client):
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": "http://localhost:11434/v1",
                "api_key": "not-needed",
                "model_name": "llama3.2:3b",
                "vision_enabled": False,
            }
        )

        r = client.get("/api/byom/state")
        assert r.json()["model"] == "llama3.2:3b"

    def test_recommendations_has_text_models(self, client):
        """Transcribe receives text-only LLM recommendations for its post-transcription endpoint."""
        r = client.get("/api/byom/state")
        recs = r.json()["recommendations"]

        for tier in ("laptop", "desktop", "mac_unified"):
            assert tier in recs
            assert len(recs[tier]) > 0, f"transcribe has no recommendations for {tier}"
            for entry in recs[tier]:
                assert entry["vision"] is False, (
                    f"transcribe {tier} recommendation {entry['model_name']!r} has vision=True"
                )
                assert "ethos_badges" in entry
                assert "role" in entry
                assert "notes" in entry

    def test_response_is_json_serialisable(self, client):
        r = client.get("/api/byom/state")
        json.dumps(r.json())

    def test_state_roles_match_role_setting(self, client):
        """state.roles is derived from _ROLE_SETTING keys — derive one from the
        other and assert equality, so the two can never drift apart."""
        from artifice_transcribe.web.routers.byom import _ROLE_SETTING

        r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["roles"] == list(_ROLE_SETTING)
        assert r.json()["roles"] == ["chat"]


# ── GET /api/byom/detect ──────────────────────────────────────────────────


class TestByomDetect:
    def test_returns_endpoints_array(self, client):
        with patch("artifice_transcribe.web.routers.byom.detect_local_servers") as mock_detect:
            mock_detect.return_value = [
                ProbeResult(
                    url="http://localhost:11434/v1",
                    reachable=True,
                    provider="ollama",
                    models=("llama3.2:3b",),
                    hint=None,
                ),
            ]

            r = client.get("/api/byom/detect")
            assert r.status_code == 200
            ep = r.json()["endpoints"][0]
            assert ep["name"] == "Ollama"
            assert ep["reachable"] is True

    def test_unreachable_endpoint_has_hint(self, client):
        with patch("artifice_transcribe.web.routers.byom.detect_local_servers") as mock_detect:
            mock_detect.return_value = [
                ProbeResult(
                    url="http://localhost:8080/v1",
                    reachable=False,
                    provider="generic-api",
                    models=(),
                    hint="Server not running",
                ),
            ]

            r = client.get("/api/byom/detect")
            ep = r.json()["endpoints"][0]
            assert ep["reachable"] is False
            assert ep["name"] == "vLLM / LocalAI"


# ── POST /api/byom/test ───────────────────────────────────────────────────


class TestByomTest:
    def test_rejects_empty_url(self, client):
        r = client.post("/api/byom/test", json={"url": "", "api_key": ""})
        assert r.status_code == 400
        body = r.json()
        assert "hint" in body
        assert "error" in body

    def test_rejects_bad_scheme(self, client):
        r = client.post("/api/byom/test", json={"url": "ftp://localhost/v1", "api_key": ""})
        assert r.status_code == 400
        body = r.json()
        assert "hint" in body
        assert "error" in body

    def test_rejects_public_url_by_default(self, client):
        r = client.post("/api/byom/test", json={"url": "https://api.example.com/v1", "api_key": ""})
        assert r.status_code == 400
        body = r.json()
        assert "hint" in body
        assert "error" in body

    def test_successful_probe_saves_config(self, client):
        with patch("artifice_transcribe.web.routers.byom.probe_endpoint") as mock_probe:
            mock_probe.return_value = ProbeResult(
                url="http://localhost:11434/v1",
                reachable=True,
                provider="ollama",
                models=("llama3.2:3b",),
                hint=None,
            )

            r = client.post(
                "/api/byom/test",
                json={
                    "url": "http://localhost:11434/v1",
                    "api_key": "sk-real",
                },
            )
            assert r.status_code == 200
            assert r.json()["reachable"] is True

            state_r = client.get("/api/byom/state")
            assert state_r.json()["configured"] is True

    def test_unreachable_probe_does_not_save(self, client):
        with patch("artifice_transcribe.web.routers.byom.probe_endpoint") as mock_probe:
            mock_probe.return_value = ProbeResult(
                url="http://localhost:1234/v1",
                reachable=False,
                provider="lm-studio",
                models=(),
                hint="Not running",
            )

            r = client.post(
                "/api/byom/test",
                json={
                    "url": "http://localhost:1234/v1",
                    "api_key": "sk-test",
                },
            )
            assert r.json()["reachable"] is False

            state_r = client.get("/api/byom/state")
            assert state_r.json()["configured"] is False

    def test_api_key_persisted_on_success(self, client):
        with patch("artifice_transcribe.web.routers.byom.probe_endpoint") as mock_probe:
            mock_probe.return_value = ProbeResult(
                url="http://localhost:9999/v1",
                reachable=True,
                provider="generic-api",
                models=(),
                hint=None,
            )

            client.post(
                "/api/byom/test",
                json={
                    "url": "http://localhost:9999/v1",
                    "api_key": "sk-secret",
                },
            )

            from artifice_transcribe.api.v1.routes import _load_inference_config

            saved = _load_inference_config()
            assert saved.get("api_key") == "sk-secret"
            assert saved.get("base_url") == "http://localhost:9999/v1"


# ── Contract + SSRF + first-paint + router-collision tests (pytest-httpx) ─


class TestByomContractAndSsrf:
    """Drive the real HTTP layer so the JSON keys match what byom.js reads."""

    # -- GET /api/byom/state ---------------------------------------------

    def test_state_json_keys_default(self, client):
        r = client.get("/api/byom/state")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "app",
            "configured",
            "endpoint",
            "model",
            "roles",
            "recommendations",
        }
        assert body["app"] == "artifice-transcribe"
        assert body["configured"] is False

    def test_state_json_keys_configured(self, client):
        from artifice_transcribe.api.v1.routes import _save_inference_config

        _save_inference_config(
            {
                "base_url": "http://localhost:9999/v1",
                "api_key": "sk-real",
                "model_name": "llama3.2:3b",
                "vision_enabled": False,
            }
        )
        r = client.get("/api/byom/state")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {
            "app",
            "configured",
            "endpoint",
            "model",
            "roles",
            "recommendations",
        }
        assert body["configured"] is True

    # -- GET /api/byom/detect --------------------------------------------

    def test_detect_json_keys_all_down(self, client, httpx_mock):
        for url in ("http://localhost:11434", "http://localhost:1234", "http://localhost:8080"):
            httpx_mock.add_exception(
                httpx.ConnectError("Connection refused"), url=url + "/api/tags"
            )

        r = client.get("/api/byom/detect")
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"endpoints"}
        assert len(body["endpoints"]) == 3
        for ep in body["endpoints"]:
            assert set(ep.keys()) == {"url", "name", "provider", "reachable", "models", "hint"}
            assert ep["reachable"] is False

    def test_detect_json_keys_one_up(self, client, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags", json={"models": [{"name": "llama3.2:3b"}]}
        )
        httpx_mock.add_response(
            url="http://localhost:11434/v1/models", json={"data": [{"id": "llama3.2:3b"}]}
        )
        for url in ("http://localhost:1234", "http://localhost:8080"):
            httpx_mock.add_exception(
                httpx.ConnectError("Connection refused"), url=url + "/api/tags"
            )

        r = client.get("/api/byom/detect")
        assert r.status_code == 200
        reachable = [ep for ep in r.json()["endpoints"] if ep["reachable"]]
        assert len(reachable) == 1
        assert reachable[0]["provider"] == "ollama"

    # -- POST /api/byom/test ---------------------------------------------

    def test_test_json_keys_reachable(self, client, httpx_mock):
        httpx_mock.add_response(
            url="http://localhost:11434/api/tags", json={"models": [{"name": "llama3.2:3b"}]}
        )
        httpx_mock.add_response(
            url="http://localhost:11434/v1/models", json={"data": [{"id": "llama3.2:3b"}]}
        )

        r = client.post("/api/byom/test", json={"url": "http://localhost:11434/v1", "api_key": ""})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"reachable", "provider", "models", "hint"}
        assert body["reachable"] is True

    def test_test_json_keys_refused(self, client, httpx_mock):
        httpx_mock.add_exception(
            httpx.ConnectError("Connection refused"), url="http://localhost:11434/api/tags"
        )

        r = client.post("/api/byom/test", json={"url": "http://localhost:11434/v1", "api_key": ""})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"reachable", "provider", "models", "hint"}
        assert body["reachable"] is False

    def test_test_json_keys_timeout(self, client, httpx_mock):
        httpx_mock.add_exception(
            httpx.TimeoutException("Timed out"), url="http://localhost:11434/api/tags"
        )

        r = client.post("/api/byom/test", json={"url": "http://localhost:11434/v1", "api_key": ""})
        assert r.status_code == 200
        body = r.json()
        assert set(body.keys()) == {"reachable", "provider", "models", "hint"}
        assert body["reachable"] is False

    # -- SSRF surface -----------------------------------------------------

    def _assert_rejected_before_network(self, r, httpx_mock):
        assert r.status_code == 400
        body = r.json()
        assert "hint" in body
        assert "error" in body
        assert isinstance(body["hint"], str)
        assert not httpx_mock.get_requests(), "a request was issued for a rejected URL"

    def test_rejects_link_local_before_network(self, client, httpx_mock):
        r = client.post(
            "/api/byom/test",
            json={"url": "http://169.254.169.254/latest/meta-data/", "api_key": ""},
        )
        self._assert_rejected_before_network(r, httpx_mock)

    def test_rejects_public_host_before_network(self, client, httpx_mock):
        r = client.post("/api/byom/test", json={"url": "https://api.openai.com/v1", "api_key": ""})
        self._assert_rejected_before_network(r, httpx_mock)

    def test_rejects_file_scheme_before_network(self, client, httpx_mock):
        r = client.post("/api/byom/test", json={"url": "file:///etc/passwd", "api_key": ""})
        self._assert_rejected_before_network(r, httpx_mock)

    def test_rejects_no_host_before_network(self, client, httpx_mock):
        r = client.post("/api/byom/test", json={"url": "http:///v1", "api_key": ""})
        self._assert_rejected_before_network(r, httpx_mock)

    # -- First paint must not block ---------------------------------------

    def test_root_does_not_probe_and_links_assets(self, client, httpx_mock):
        for url in ("http://localhost:11434", "http://localhost:1234", "http://localhost:8080"):
            httpx_mock.add_exception(
                httpx.ConnectError("Connection refused"), url=url + "/api/tags", is_optional=True
            )
        for url in (
            "http://localhost:11434/v1",
            "http://localhost:1234/v1",
            "http://localhost:8080/v1",
        ):
            httpx_mock.add_exception(
                httpx.ConnectError("Connection refused"), url=url + "/models", is_optional=True
            )

        start = time.perf_counter()
        r = client.get("/")
        elapsed = time.perf_counter() - start

        assert r.status_code == 200
        assert elapsed < 0.5, f"GET / took {elapsed:.3f}s, root must not probe"
        assert not httpx_mock.get_requests(), "root route made a network request"
        html = r.text
        assert "/shared/byom.css" in html
        assert "/shared/byom.js" in html
        assert "/shared/shell.css" in html
        assert "app-shell" in html
        assert "global-search-input" in html

    def test_about_page_serves(self, client):
        r = client.get("/about")
        assert r.status_code == 200
        assert "About ArtificeTranscribe" in r.text
        assert "app-shell" in r.text

    def test_static_index_html_is_gone(self, client):
        r = client.get("/static/index.html")
        assert r.status_code == 404


class TestTranscribeRouterCollision:
    """Transcribe mounts /api/v1 and /api/byom side by side; neither must shadow the other."""

    def test_api_v1_jobs_still_resolves(self, client):
        r = client.get("/api/v1/jobs")
        assert r.status_code == 200
        # Empty list is the expected body for a fresh database.
        assert r.json() == []

    def test_api_v1_health_detailed_still_resolves(self, client):
        r = client.get("/api/v1/health/detailed")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body or "healthy" in body

    def test_api_byom_state_resolves(self, client):
        r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["app"] == "artifice-transcribe"


class TestRecommendationsGuard:
    """The ``try/except KeyError`` guard is defensive code for any unregistered app.

    Since transcribe now has registry entries, the KeyError path is never hit in
    normal operation, but it must still work correctly when deliberately triggered.
    """

    def test_transcribe_recommendations_are_non_empty(self, client):
        r = client.get("/api/byom/state")
        assert r.status_code == 200
        recs = r.json()["recommendations"]
        for tier in ("laptop", "desktop", "mac_unified"):
            assert tier in recs
            assert len(recs[tier]) > 0, f"transcribe should have recommendations for {tier}"

    def test_keyerror_returns_empty_recommendations(self, client):
        # recommendations_for_app now runs inside model_harness.byom.byom_recommendations
        # (the app's _byom_recommendations is just an alias for it since the PR6 dedup),
        # so the guard is exercised end-to-end through the real HTTP endpoint here,
        # patched at the module that actually calls it.
        with patch("model_harness.byom.recommendations_for_app") as mock_recs:
            mock_recs.side_effect = KeyError("artifice-transcribe")
            r = client.get("/api/byom/state")
        assert r.status_code == 200
        assert r.json()["recommendations"] == {
            "laptop": [],
            "desktop": [],
            "mac_unified": [],
        }

    def test_non_keyerror_propagates(self, client):
        with patch("model_harness.byom.recommendations_for_app") as mock_recs:
            mock_recs.side_effect = ValueError("unexpected failure")
            with pytest.raises(ValueError, match="unexpected failure"):
                client.get("/api/byom/state")

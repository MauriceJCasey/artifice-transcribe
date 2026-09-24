# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Playwright harness for the deterministic Transcribe browser baseline.

``TranscribeUi`` drives a real Chromium page against the real FastAPI app (which
serves the Jinja templates and static assets) while intercepting every
``/api/**`` request with the synthetic fixtures in ``_fixtures.py``.  It records
unexpected API requests, browser page errors, console errors and same-origin
error responses so each journey can fail on them at the end.

The harness is intentionally stateful in a small, legible way: the upload,
segment-edit, speaker-rename, metadata and export handlers mutate an in-memory
transcript so that an edit made through the real UI flows into the export
content — the payload/content assertion Stage 0 requires.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Browser, BrowserContext, Page, expect

from . import _fixtures as fx

_CONTENT_TYPES = {
    "json": "application/json",
    "srt": "text/srt",
    "vtt": "text/vtt",
    "txt": "text/plain",
    "md": "text/markdown",
    "pdf": "application/pdf",
    "ohms": "application/xml",
    "tei": "application/xml",
}


class TranscribeUi:
    """One browser context with synthetic, stateful API fixtures."""

    def __init__(
        self,
        context: BrowserContext,
        page: Page,
        base_url: str,
        *,
        asr_available: bool = True,
        byom_configured: bool = False,
        model_keys: list[str] | None = None,
        populated: bool = True,
        capture_root: Path | None = None,
    ) -> None:
        self.context = context
        self.page = page
        self.base_url = base_url
        self.asr_available = asr_available
        self.byom_configured = byom_configured
        self.model_keys = model_keys if model_keys is not None else list(fx.ASR_MODEL_KEYS)
        self.populated = populated
        self.capture_root = capture_root or Path(
            os.environ.get("ARTIFICE_TRANSCRIBE_UI_ARTIFACTS", ".artifacts/ui-redesign/transcribe")
        )

        # Synthetic, mutable server-side state.
        self.job = fx.job()
        self._segments = fx.transcript()["segments"]
        self._speakers = fx.speakers()["speakers"]

        # Recorded interactions, asserted at the end of each journey.
        self.seen: list[str] = []
        self.unexpected: list[str] = []
        self.page_errors: list[str] = []
        self.console_errors: list[str] = []
        self.bad_responses: list[str] = []
        self.uploads: list[dict] = []
        self.segment_patches: list[dict] = []
        self.speaker_patches: list[dict] = []
        self.metadata_patches: list[dict] = []
        self.exports: list[dict] = []
        self.consents: list[dict] = []
        self.downloads: list[str] = []

        # Explicitly permitted same-origin failures: "METHOD /path" strings that
        # a journey deliberately provokes (e.g. a 409 for an in-progress export).
        # Anything not listed here and >=400 fails ``assert_clean``.
        self.expected_failures: set[str] = set()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    @classmethod
    def launch(
        cls,
        browser: Browser,
        base_url: str,
        *,
        asr_available: bool = True,
        byom_configured: bool = False,
        model_keys: list[str] | None = None,
        populated: bool = True,
    ) -> TranscribeUi:
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        context.set_default_timeout(5000)
        page = context.new_page()
        ui = cls(
            context,
            page,
            base_url,
            asr_available=asr_available,
            byom_configured=byom_configured,
            model_keys=model_keys,
            populated=populated,
        )
        ui._wire()
        ui.goto("/")
        ui.wait_ready()
        return ui

    def close(self) -> None:
        self.context.close()

    def _wire(self) -> None:
        self.page.on("pageerror", lambda error: self.page_errors.append(str(error)))
        self.page.on("console", self._on_console)
        self.page.on("response", self._on_response)
        self.page.on("dialog", lambda dialog: dialog.accept())
        self.context.route("**/api/**", self._on_route)

    # ── Navigation & helpers ─────────────────────────────────────────────────

    def goto(self, path: str) -> None:
        self.page.goto(f"{self.base_url}{path}", wait_until="domcontentloaded")

    def wait_ready(self) -> None:
        # byom.js autostart resolves /api/byom/state and stamps data-state on the
        # masthead model control — the last first-load signal to settle.
        expect(self.page.locator('[data-shell-action="model"]')).to_have_attribute(
            "data-state", re.compile("^(configured|unconfigured)$"), timeout=15_000
        )

    def tab(self, name: str) -> None:
        # The visible navigation lives in the shared shell's titlebar
        # (.shell-titlebar-nav, single-line research shell); the app's
        # legacy .tab buttons remain the panel controller but are visually
        # clipped by shared CSS (see UI_REDESIGN_PLAN.md §2). Clicking a
        # titlebar nav link navigates to /?view=… and reloads the page.
        with self.page.expect_navigation(wait_until="domcontentloaded"):
            self.page.locator(f'.shell-titlebar-nav a[href="/?view={name}"]').click()
        expect(self.page.locator(f"#panel-{name}")).to_be_visible()

    def wait_until(self, predicate, timeout_ms: int = 5000) -> bool:
        """Pump the Playwright loop until *predicate* returns truthy."""
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            if predicate():
                return True
            self.page.wait_for_timeout(50)
        return False

    def set_theme(self, theme: str) -> None:
        self.page.evaluate(
            "(theme) => { document.documentElement.setAttribute('data-theme', theme); "
            "localStorage.setItem('pt-theme', theme); }",
            theme,
        )
        self.page.wait_for_timeout(120)

    def screenshot(self, name: str) -> None:
        """Capture the current view in both themes under the gitignored artifact dir."""
        self.capture_root.mkdir(parents=True, exist_ok=True)
        for theme in ("light", "dark"):
            self.set_theme(theme)
            self.page.screenshot(
                path=str(self.capture_root / f"{name}-{theme}.png"), full_page=True
            )

    def assert_clean(self) -> None:
        assert not self.page_errors, f"browser page errors: {self.page_errors}"
        assert not self.console_errors, f"console errors: {self.console_errors}"
        assert not self.unexpected, f"unexpected API requests: {self.unexpected}"
        assert not self.bad_responses, f"unexpected error responses: {self.bad_responses}"

    # ── Event listeners ──────────────────────────────────────────────────────

    def _on_console(self, message) -> None:
        if message.type == "error":
            self.console_errors.append(message.text)

    def _on_response(self, response) -> None:
        if response.status < 400 or not response.url.startswith(self.base_url):
            return
        path = urlparse(response.url).path
        key = f"{response.request.method} {path}"
        if key in self.expected_failures:
            return
        self.bad_responses.append(f"{response.status} {response.url}")

    # ── Route dispatch ───────────────────────────────────────────────────────

    def _on_route(self, route, request) -> None:
        method = request.method
        parsed = urlparse(request.url)
        path = parsed.path
        query = parsed.query
        key = f"{method} {path}"
        self.seen.append(key)

        handler = self._lookup(method, path)
        if handler is None:
            self.unexpected.append(key)
            route.abort()
            return
        handler(route, request, query)

    def _lookup(self, method: str, path: str):
        if path == "/api/ui/preferences" and method == "GET":
            return self._handle_preferences
        if path == "/api/suite/apps" and method == "GET":
            return self._handle_suite_apps
        if path == "/api/byom/state" and method == "GET":
            return self._handle_byom_state
        if path == "/api/byom/detect" and method == "GET":
            return self._handle_byom_detect
        if path == "/api/v1/capabilities" and method == "GET":
            return self._handle_capabilities
        if path == "/api/v1/config" and method == "GET":
            return self._handle_config_get
        if path == "/api/v1/config" and method == "PATCH":
            return self._handle_config_patch
        if path == "/api/v1/health/detailed" and method == "GET":
            return self._handle_health
        if path == "/api/v1/jobs" and method == "GET":
            return self._handle_jobs_list
        if path == "/api/v1/dictionary" and method == "GET":
            return self._handle_dictionary
        if path == "/api/v1/speakers/known" and method == "GET":
            return self._handle_known_speakers
        if path == "/api/v1/models" and method == "GET":
            return self._handle_models_list
        if path == "/api/v1/transcribe" and method == "POST":
            return self._handle_upload
        if path == f"/api/v1/jobs/{fx.JOB_ID}" and method == "GET":
            return self._handle_job_get
        if path == f"/api/v1/jobs/{fx.JOB_ID}/transcript" and method == "GET":
            return self._handle_transcript
        if path == f"/api/v1/jobs/{fx.JOB_ID}/speakers" and method == "GET":
            return self._handle_speakers_get
        if path == f"/api/v1/jobs/{fx.JOB_ID}/speakers" and method == "PATCH":
            return self._handle_speakers_patch
        if path == f"/api/v1/jobs/{fx.JOB_ID}/segments" and method == "PATCH":
            return self._handle_segments_patch
        if path == f"/api/v1/jobs/{fx.JOB_ID}/metadata" and method == "PATCH":
            return self._handle_metadata_patch
        if path == f"/api/v1/jobs/{fx.JOB_ID}/audio" and method == "GET":
            return self._handle_audio
        if path == f"/api/v1/jobs/{fx.JOB_ID}/export" and method == "GET":
            return self._handle_export

        match = re.fullmatch(r"/api/v1/models/([^/]+)", path)
        if match and method == "GET":
            key = match.group(1)
            return lambda route, request, query: self._handle_model_info(key, route)
        match = re.fullmatch(r"/api/v1/models/([^/]+)/consent", path)
        if match and method == "POST":
            key = match.group(1)
            return lambda route, request, query: self._handle_consent(key, route, request)
        match = re.fullmatch(r"/api/v1/models/([^/]+)/download", path)
        if match and method == "POST":
            key = match.group(1)
            return lambda route, request, query: self._handle_download(key, route)
        match = re.fullmatch(r"/api/v1/models/([^/]+)/download/progress", path)
        if match and method == "GET":
            key = match.group(1)
            return lambda route, request, query: self._handle_progress(key, route)
        return None

    # ── Static / read-only handlers ──────────────────────────────────────────

    def _json(self, route, payload, status: int = 200) -> None:
        route.fulfill(
            status=status,
            headers={"content-type": "application/json"},
            body=json.dumps(payload),
        )

    def _handle_preferences(self, route, request, query) -> None:
        self._json(route, {"theme": "system", "reduced_motion": False})

    def _handle_suite_apps(self, route, request, query) -> None:
        self._json(route, [])

    def _handle_byom_state(self, route, request, query) -> None:
        self._json(route, fx.byom_state(self.byom_configured))

    def _handle_byom_detect(self, route, request, query) -> None:
        self._json(route, fx.byom_detect())

    def _handle_capabilities(self, route, request, query) -> None:
        self._json(route, fx.capabilities(self.asr_available))

    def _handle_config_get(self, route, request, query) -> None:
        self._json(route, fx.model_config())

    def _handle_config_patch(self, route, request, query) -> None:
        body = json.loads(request.post_data or "{}")
        self._json(route, {"status": "updated", "changes": list(body.keys())})

    def _handle_health(self, route, request, query) -> None:
        self._json(route, fx.health_detailed())

    def _handle_jobs_list(self, route, request, query) -> None:
        self._json(route, [self.job] if self.populated else [])

    def _handle_dictionary(self, route, request, query) -> None:
        self._json(route, None)

    def _handle_known_speakers(self, route, request, query) -> None:
        self._json(route, {"speakers": []})

    def _handle_models_list(self, route, request, query) -> None:
        self._json(route, {"models": [fx.model_summary(k) for k in self.model_keys]})

    def _handle_model_info(self, key: str, route) -> None:
        self._json(route, fx.model_summary(key))

    def _handle_job_get(self, route, request, query) -> None:
        self._json(route, self.job)

    def _name_map(self) -> dict[str, str]:
        return {sp["speaker_label"]: sp["custom_name"] for sp in self._speakers}

    def _mapped_segments(self) -> list[dict]:
        name_map = self._name_map()
        return [
            {**seg, "speaker_label": name_map.get(seg["speaker_label"], seg["speaker_label"])}
            for seg in self._segments
        ]

    def _handle_transcript(self, route, request, query) -> None:
        self._json(route, {"job_id": fx.JOB_ID, "segments": self._mapped_segments()})

    def _handle_speakers_get(self, route, request, query) -> None:
        self._json(route, {"job_id": fx.JOB_ID, "speakers": self._speakers})

    def _handle_audio(self, route, request, query) -> None:
        route.fulfill(
            status=200,
            headers={"content-type": "audio/wav"},
            body=fx.silent_wav(),
        )

    # ── Stateful handlers ────────────────────────────────────────────────────

    def _handle_upload(self, route, request, query) -> None:
        self.uploads.append({"query": query, "url": request.url})
        self._json(
            route,
            {"job_id": fx.JOB_ID, "status": "queued", "message": "Transcription job accepted"},
            status=202,
        )

    def _handle_segments_patch(self, route, request, query) -> None:
        body = json.loads(request.post_data or "{}")
        updates = body.get("updates", [])
        self.segment_patches.extend(updates)
        for update in updates:
            seg_id = update.get("segment_id")
            text = update.get("text")
            if seg_id and text is not None:
                for seg in self._segments:
                    if seg["id"] == seg_id:
                        seg["text"] = text
        self._json(route, {"job_id": fx.JOB_ID, "updated_count": len(updates)})

    def _handle_speakers_patch(self, route, request, query) -> None:
        body = json.loads(request.post_data or "{}")
        self.speaker_patches.append(body)
        for rename in body.get("speakers", []):
            for speaker in self._speakers:
                if speaker["speaker_label"] == rename.get("speaker_label"):
                    speaker["custom_name"] = rename.get("custom_name")
        self._json(route, {"job_id": fx.JOB_ID, "speakers": self._speakers})

    def _handle_metadata_patch(self, route, request, query) -> None:
        body = json.loads(request.post_data or "{}")
        self.metadata_patches.append(body)
        self.job.update(body)
        self._json(route, self.job)

    def _handle_export(self, route, request, query) -> None:
        fmt = parse_qs(query).get("format", ["json"])[0]
        if fmt == "json":
            payload = {"job_id": fx.JOB_ID, "segments": self._mapped_segments()}
            self.exports.append({"format": fmt, "body": json.dumps(payload)})
            self._json(route, payload)
            return
        body = fx.export_txt_body(self._segments, self._name_map())
        self.exports.append({"format": fmt, "body": body})
        route.fulfill(
            status=200,
            headers={"content-type": _CONTENT_TYPES.get(fmt, "application/octet-stream")},
            body=body,
        )

    def _handle_consent(self, key: str, route, request) -> None:
        body = json.loads(request.post_data or "{}")
        self.consents.append({"key": key, "consent": bool(body.get("consent"))})
        self._json(route, {"key": key, "consented": bool(body.get("consent"))})

    def _handle_download(self, key: str, route) -> None:
        self.downloads.append(key)
        self._json(route, fx.download_started(key))

    def _handle_progress(self, key: str, route) -> None:
        # A heartbeat-only SSE so the dialog's EventSource ends without the
        # production "completed" message (whose close/reopen race is out of
        # scope for this baseline). The download POST assertion is the contract.
        body = f"data: {json.dumps({'type': 'heartbeat', 'key': key})}\n\n"
        route.fulfill(status=200, headers={"content-type": "text/event-stream"}, body=body)

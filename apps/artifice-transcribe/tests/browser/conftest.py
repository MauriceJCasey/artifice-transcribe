# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Fixtures for the deterministic Transcribe browser baseline.

Serves the real FastAPI app (Jinja templates + static assets) on an ephemeral
loopback port with the lifespan disabled, so no database is created and no ASR
stack is imported.  Every ``/api/**`` request is intercepted by the
:class:`~._harness.TranscribeUi` harness and answered with synthetic fixtures.
"""

from __future__ import annotations

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from playwright.sync_api import sync_playwright

from ._harness import TranscribeUi


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def transcribe_server(tmp_path_factory):
    """Serve the real FastAPI app on an ephemeral loopback port, lifespan off."""
    from artifice_transcribe import config

    # Defensive isolation: point the upload/output roots at a throwaway dir so
    # that even a request which escaped route interception could not touch real
    # user data.  The database engine is never connected (lifespan="off" and all
    # /api traffic is intercepted), so it is deliberately left alone.
    data_dir = tmp_path_factory.mktemp("transcribe-ui")
    config.settings.upload_dir = str(data_dir / "uploads")
    config.settings.output_dir = str(data_dir / "outputs")

    from artifice_transcribe.main import app  # noqa: PLC0415  (imported after settings)

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, lifespan="off", log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=0.25).status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=2)
        pytest.fail("isolated Transcribe UI server did not start")

    yield base_url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="session")
def chromium_browser():
    """Share one browser process; every journey isolates itself in a context."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
        yield browser
        browser.close()


@pytest.fixture
def ui(transcribe_server, chromium_browser):
    """Return a factory that builds an isolated :class:`TranscribeUi`.

    Usage: ``u = ui(asr_available=False, byom_configured=True, model_keys=[...])``.
    Every instance is closed at test teardown.
    """
    instances: list[TranscribeUi] = []

    def make(**kwargs) -> TranscribeUi:
        instance = TranscribeUi.launch(chromium_browser, transcribe_server, **kwargs)
        instances.append(instance)
        return instance

    yield make

    for instance in instances:
        instance.close()

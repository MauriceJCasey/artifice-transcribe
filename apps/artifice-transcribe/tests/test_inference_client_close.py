# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression tests for the AsyncOpenAI client leak in the inference routes.

``InferenceEngine.__init__`` builds an ``AsyncOpenAI`` — which wraps its own
``httpx.AsyncClient`` connection pool — and the three inference route handlers
used to construct a fresh engine per request without ever closing it.  Under
load (e.g. summarising many completed jobs back-to-back) that leaks TCP
sockets until GC reclaims them.

These tests prove the engine is closed on the happy path *and* on the error
path, driving the real ASGI app via the ``api`` fixture so the assertion
covers the route wiring, not just the ``aclose`` method in isolation.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio

_FAKE_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "not-needed",
    "model_name": "",
    "vision_enabled": False,
}


def _patch_config(monkeypatch) -> None:
    """Point the inference config load at a fixed in-memory dict so no test
    touches the real on-disk config (or runs its legacy migration)."""
    monkeypatch.setattr(
        "artifice_transcribe.api.v1.routes._load_inference_config",
        lambda: dict(_FAKE_CONFIG),
    )


def _install_fake_generate(monkeypatch, fake_generate):
    """Monkeypatch ``InferenceEngine.generate`` (returns no real endpoint call)
    and ``InferenceEngine.aclose`` (records the close) on the class the routes
    import, returning the list that receives close records."""
    close_calls: list[bool] = []

    async def _record_close(self) -> None:
        close_calls.append(True)

    monkeypatch.setattr("artifice_transcribe.api.v1.routes.InferenceEngine.generate", fake_generate)
    monkeypatch.setattr("artifice_transcribe.api.v1.routes.InferenceEngine.aclose", _record_close)
    return close_calls


async def _create_completed_job_with_segment(api, job_id: str) -> None:
    """Insert a completed transcription job with one segment."""
    from artifice_transcribe.db.models import JobStatus, TranscriptionJob, TranscriptSegment

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


async def test_aclose_closes_underlying_client(monkeypatch):
    """``InferenceEngine.aclose`` awaits the wrapped client's ``close``."""
    from artifice_transcribe.services.inference import InferenceEngine

    engine = InferenceEngine()

    class StubClient:
        def __init__(self):
            self.closed = False

        async def close(self):
            self.closed = True

    stub = StubClient()
    monkeypatch.setattr(engine, "client", stub)

    await engine.aclose()

    assert stub.closed is True


async def test_nonstreaming_generate_closes_engine(api, monkeypatch):
    """``inference_generate`` with ``stream=False`` closes the engine after the
    awaited call returns."""

    async def fake_generate(
        self, prompt, image_base64=None, stream=False, temperature=0.2, max_tokens=2048
    ):
        return "generated response"

    _patch_config(monkeypatch)
    close_calls = _install_fake_generate(monkeypatch, fake_generate)

    resp = await api.client.post(
        "/api/v1/inference/generate",
        json={"prompt": "Hello", "stream": False},
    )

    assert resp.status_code == 200
    assert resp.json() == {"response": "generated response"}
    assert close_calls == [True]


async def test_nonstreaming_generate_closes_on_error(api, monkeypatch):
    """If ``generate`` raises on the non-streaming path, the ``finally`` still
    closes the engine before the exception propagates."""

    async def fake_generate(
        self, prompt, image_base64=None, stream=False, temperature=0.2, max_tokens=2048
    ):
        raise RuntimeError("boom")

    _patch_config(monkeypatch)
    close_calls = _install_fake_generate(monkeypatch, fake_generate)

    with pytest.raises(RuntimeError, match="boom"):
        await api.client.post(
            "/api/v1/inference/generate",
            json={"prompt": "Hello", "stream": False},
        )

    assert close_calls == [True]


async def test_streaming_generate_closes_after_drain(api, monkeypatch):
    """``inference_generate`` with ``stream=True`` closes the engine inside the
    generator *after* the stream is drained — not before it starts."""

    async def fake_generate(
        self, prompt, image_base64=None, stream=False, temperature=0.2, max_tokens=2048
    ):
        async def _gen():
            yield "chunk-a"
            yield "chunk-b"

        return _gen()

    _patch_config(monkeypatch)
    close_calls = _install_fake_generate(monkeypatch, fake_generate)

    resp = await api.client.post(
        "/api/v1/inference/generate",
        json={"prompt": "Hello", "stream": True},
    )

    assert resp.status_code == 200
    assert resp.text == "chunk-achunk-b"
    assert close_calls == [True]


async def test_summarize_closes_engine_on_error(api, monkeypatch):
    """``summarize_job`` closes the engine when ``generate`` raises, and the
    SSE stream still ends with the ``done`` event."""

    async def fake_generate(
        self, prompt, image_base64=None, stream=False, temperature=0.2, max_tokens=2048
    ):
        raise RuntimeError("boom")

    _patch_config(monkeypatch)
    close_calls = _install_fake_generate(monkeypatch, fake_generate)
    await _create_completed_job_with_segment(api, "job-close-sum")

    resp = await api.client.post("/api/v1/jobs/job-close-sum/summarize")

    assert resp.status_code == 200
    assert '"type": "error"' in resp.text
    assert '"type": "done"' in resp.text
    assert close_calls == [True]


async def test_cleanup_closes_engine_on_error(api, monkeypatch):
    """``cleanup_job`` closes the engine when ``generate`` raises."""

    async def fake_generate(
        self, prompt, image_base64=None, stream=False, temperature=0.2, max_tokens=2048
    ):
        raise RuntimeError("boom")

    _patch_config(monkeypatch)
    close_calls = _install_fake_generate(monkeypatch, fake_generate)
    await _create_completed_job_with_segment(api, "job-close-cln")

    resp = await api.client.post("/api/v1/jobs/job-close-cln/cleanup")

    assert resp.status_code == 200
    assert '"type": "error"' in resp.text
    assert '"type": "done"' in resp.text
    assert close_calls == [True]

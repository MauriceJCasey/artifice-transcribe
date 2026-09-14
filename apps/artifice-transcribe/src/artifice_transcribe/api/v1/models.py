# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import json
import logging
from queue import Empty

import model_harness.registry as reg
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from artifice_transcribe.api.v1.routes import _load_hf_token
from artifice_transcribe.schemas.transcription import (
    ConsentRequest,
    ConsentResponse,
    DownloadStartResponse,
    ModelDownloadInfo,
    ModelInfoResponse,
    ModelListResponse,
)
from artifice_transcribe.services.download import (
    find_registry_key,
    get_download_manager,
    hf_cache_dir,
    human_size,
    is_consented,
    record_consent,
    requires_token,
    resolve_transitive,
    revoke_consent,
    total_transitive_size,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["models"])


# ── ASR Model Download ───────────────────────────────────────────────────────
#
# These endpoints support the consent-and-download flow for multi-gigabyte ASR
# model weights.  The dialog itself is rendered by the UI layer; these endpoints
# provide the data it needs: model inventory, transitive size disclosure,
# on-disk destination, consent persistence, and real-time progress via SSE.
#
# No endpoint imports ``torch`` at module scope — the lightweight install
# (no ``--extra asr``) serves the model-list and consent endpoints so a user
# can discover what is available before installing anything.


def _model_info_response(key: str) -> ModelInfoResponse:
    """Build a :class:`ModelInfoResponse` from the registry."""
    models = resolve_transitive(key)
    total = total_transitive_size(key)
    need_token = requires_token(key)

    # Build the list with the registry key for each entry.
    model_list: list[ModelDownloadInfo] = []
    for m in models:
        m_key = find_registry_key(m)
        model_list.append(
            ModelDownloadInfo(
                key=m_key,
                hf_repo=m.hf_repo,
                size_bytes=m.size_bytes,
                size_human=human_size(m.size_bytes),
                requires_hf_token=m.requires_hf_token,
                description=m.description,
            )
        )

    return ModelInfoResponse(
        key=key,
        models=model_list,
        total_size_bytes=total,
        total_size_human=human_size(total),
        requires_hf_token=need_token,
        cache_directory=str(hf_cache_dir()),
        consented=is_consented(key),
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """Return all available ASR models with transitive sizes and consent state.

    This is a cheap, read-only endpoint — no imports of the ASR stack, no
    network calls, no file I/O beyond reading the consent file.
    """
    return ModelListResponse(models=[_model_info_response(key) for key in reg.ASR_MODELS])


@router.get("/models/{key}", response_model=ModelInfoResponse)
async def model_info(key: str) -> ModelInfoResponse:
    """Return detailed download info for a single model, including its
    dependencies, total transitive size, on-disk destination, and consent state.
    """
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")
    return _model_info_response(key)


@router.post("/models/{key}/consent", response_model=ConsentResponse)
async def grant_model_consent(key: str, body: ConsentRequest) -> ConsentResponse:
    """Record or revoke the user's consent to download *key*.

    Consent is persisted to ``platformdirs`` user data (not the package
    directory) so it survives reinstalls.  The download endpoint refuses to
    start without recorded consent.

    If *body.consent* is ``False``, consent is revoked.
    """
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")

    if body.consent:
        record_consent(key)
    else:
        revoke_consent(key)

    return ConsentResponse(key=key, consented=body.consent)


@router.post("/models/{key}/download", response_model=DownloadStartResponse)
async def start_model_download(key: str) -> DownloadStartResponse:
    """Begin downloading *key* and all its transitive dependencies.

    **Preconditions:**
    - Consent must have been recorded via ``POST /models/{key}/consent``.
    - If any model requires an HF token, ``hf_token`` must be set in the
      config (see ``PATCH /api/v1/config``).

    **Response:** ``202 Accepted`` with download metadata.  Progress is
    streamed via SSE at ``GET /models/{key}/download/progress``.
    """
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")

    need_token = requires_token(key)
    hf_token = _load_hf_token()

    if need_token and not hf_token:
        raise HTTPException(
            400,
            f"Model '{key}' requires a Hugging Face access token but "
            f"no HF_TOKEN is configured.  Set it via PATCH /api/v1/config "
            f"or the web UI settings panel.",
        )

    manager = get_download_manager()

    # The manager's start_download is lock-guarded and handles dedup — this
    # route is a thin caller.
    try:
        ds = manager.start_download(key, token=hf_token)
    except PermissionError:
        logger.exception("Download permission denied for model key=%s", key)
        raise HTTPException(403, "Consent has not been granted for this model download") from None

    total = total_transitive_size(key)
    return DownloadStartResponse(
        key=key,
        model_count=len(ds.models),
        total_size_bytes=total,
        total_size_human=human_size(total),
    )


@router.get("/models/{key}/download/status")
async def get_download_status(key: str) -> dict:
    """Poll the current status of a download (or lack thereof).

    Returns ``null`` state if no download has ever been started for *key*.
    For real-time progress use ``GET /models/{key}/download/progress`` (SSE).
    """
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")

    manager = get_download_manager()
    ds = manager.get_status(key)

    if ds is None:
        return {"key": key, "status": "never_started"}

    return {
        "key": key,
        "started": ds.started,
        "finished": ds.finished,
        "error_message": ds.error_message,
        "models": [
            {
                "key": ms.key,
                "hf_repo": ms.hf_repo,
                "state": ms.state.value,
                "total_bytes": ms.total_bytes,
                "downloaded_bytes": ms.downloaded_bytes,
                "error_message": ms.error_message,
            }
            for ms in ds.models
        ],
    }


@router.post("/models/{key}/download/cancel")
async def cancel_model_download(key: str) -> dict:
    """Request cancellation of an in-flight download for *key*."""
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")

    manager = get_download_manager()
    manager.cancel_download(key)
    return {"key": key, "status": "cancellation_requested"}


@router.get("/models/{key}/download/progress")
async def stream_download_progress(key: str) -> StreamingResponse:
    """SSE stream of download progress events for *key*.

    Returns an error event immediately if no download is active for *key*.
    Otherwise streams JSON-encoded events of type ``progress``, ``error``,
    ``cancelled``, ``cancelling``, or ``completed``.

    Events are formatted as standard SSE:
    ``data: {json}\\n\\n``
    """
    if key not in reg.ASR_MODELS:
        raise HTTPException(404, f"Unknown model key: {key!r}")

    manager = get_download_manager()
    ds = manager.get_status(key)

    if ds is None:

        async def _error_gen():
            payload = {"type": "error", "error": f"No download active for {key}"}
            yield f"data: {json.dumps(payload)}\n\n"

        return StreamingResponse(_error_gen(), media_type="text/event-stream")

    queue = manager.subscribe_events(key)

    async def _event_generator():
        try:
            while True:
                # Send any queued events first.
                try:
                    while True:
                        event = queue.get_nowait()
                        yield f"data: {json.dumps(event)}\n\n"
                except Empty:
                    pass

                # Check if the download is done.
                ds_now = manager.get_status(key)
                if ds_now is not None and ds_now.finished:
                    # Drain any last events.
                    while True:
                        try:
                            event = queue.get_nowait()
                            yield f"data: {json.dumps(event)}\n\n"
                        except Empty:
                            break
                    return

                # Wait for the next event.
                try:
                    event = await asyncio.to_thread(queue.get, True, 1.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except Empty:
                    # Timeout — send a heartbeat so the connection stays alive.
                    yield f"data: {json.dumps({'type': 'heartbeat', 'key': key})}\n\n"
        finally:
            manager.unsubscribe_events(key, queue)

    return StreamingResponse(_event_generator(), media_type="text/event-stream")

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
import importlib.util

from fastapi import APIRouter
from sqlalchemy import select

from artifice_transcribe.api.v1 import routes as _routes
from artifice_transcribe.api.v1.routes import _INSTALL_HINT, AsrUnavailable
from artifice_transcribe.db.models import TranscriptionJob

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health/detailed")
async def health_detailed():
    """Full health check: model load state, GPU info, database connectivity."""
    try:
        engine = _routes._get_engine()
        engine_status = engine.health_check()
    except AsrUnavailable:
        engine_status = {
            "available": False,
            "reason": str(AsrUnavailable()),
            "install_hint": _INSTALL_HINT,
        }

    db_ok = True
    try:
        async with _routes.async_session() as db:
            await db.execute(select(TranscriptionJob).limit(1))
    except Exception:
        db_ok = False

    engine_ok = engine_status.get("available") is not False

    return {
        "status": "ok" if (db_ok and engine_ok) else "degraded",
        "engine": engine_status,
        "database": {"status": "ok" if db_ok else "error"},
    }


@router.post("/health/preload")
async def health_preload():
    """Load all models into memory. Returns success or error details."""
    try:
        engine = _routes._get_engine()
        result = await asyncio.to_thread(engine.preload)
        return result
    except AsrUnavailable as exc:
        return {"ok": False, "error": exc.public_message, "install_hint": _INSTALL_HINT}


@router.get("/capabilities")
async def capabilities():
    """Report which optional features are available — never imports the ASR
    stack to answer, so this is safe and fast on a base install.

    Checks for the packages actually required by the transcription engine
    rather than using ``torch`` alone as a proxy.  A partial install where
    torch is present but whisperx (or its diarization dependency,
    pyannote.audio) is missing correctly reports unavailable.
    """
    _required = ("whisperx", "torch", "torchaudio", "torchvision", "torchcodec")
    asr_available = all(importlib.util.find_spec(pkg) is not None for pkg in _required)
    if asr_available:
        asr_info = {"available": True}
    else:
        asr_info = {
            "available": False,
            "reason": "The transcription stack is not installed.",
            "install_hint": _INSTALL_HINT,
        }
    return {"asr": asr_info}

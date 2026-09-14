# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from artifice_transcribe.api.v1 import routes as _routes
from artifice_transcribe.api.v1.routes import (
    _ASR_BACKENDS,
    AsrUnavailable,
    _load_hf_token,
    _save_hf_token,
    _save_inference_config,
    _validate_base_url,
)
from artifice_transcribe.config import settings
from artifice_transcribe.schemas.transcription import (
    InferenceConfigRequest,
    InferenceGenerateRequest,
    InferenceModelsRequest,
    InferenceTestRequest,
    ModelConfigRequest,
    ModelConfigResponse,
)
from artifice_transcribe.services.inference import (
    InferenceEngine,
    get_available_models,
)
from artifice_transcribe.services.inference import (
    test_connection as test_inf_conn,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["config"])


def _redact_model_config(key: str, value: str) -> str:
    """Return the placeholder if *key* holds a secret that is set."""
    if key in _REDACTED_MODEL_CONFIG_KEYS and value:
        return _REDACTED_PLACEHOLDER
    return value


@router.get("/config", response_model=ModelConfigResponse)
async def get_config():
    fields = {
        "whisper_model": settings.whisper_model,
        "asr_backend": settings.asr_backend,
        "device": settings.device,
        "hf_token": _load_hf_token(),
        "diarization_provider": settings.diarization_provider,
        "diarization_model": settings.diarization_model,
        "enable_alignment_model_cache": settings.enable_alignment_model_cache,
        "whisper_initial_prompt": settings.whisper_initial_prompt,
    }
    return {k: _redact_model_config(k, v) for k, v in fields.items()}


@router.patch("/config")
async def update_config(body: ModelConfigRequest):
    """Update model configuration dynamically (non-Swagger endpoint)."""
    updates = body.model_dump(exclude_unset=True)

    if "whisper_model" in updates:
        settings.whisper_model = updates["whisper_model"]
    if "asr_backend" in updates:
        if updates["asr_backend"] not in _ASR_BACKENDS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown ASR backend: {updates['asr_backend']!r}. "
                f"Choose one of {sorted(_ASR_BACKENDS)}.",
            )
        settings.asr_backend = updates["asr_backend"]
    if "device" in updates:
        settings.device = updates["device"]
    if "hf_token" in updates and updates["hf_token"] != _REDACTED_PLACEHOLDER:
        _save_hf_token(updates["hf_token"])
    if "diarization_provider" in updates:
        settings.diarization_provider = updates["diarization_provider"]
    if "diarization_model" in updates:
        settings.diarization_model = updates["diarization_model"]
    if "enable_alignment_model_cache" in updates:
        settings.enable_alignment_model_cache = updates["enable_alignment_model_cache"]
    if "whisper_initial_prompt" in updates:
        settings.whisper_initial_prompt = updates["whisper_initial_prompt"]

    if (
        "whisper_model" in updates
        or "asr_backend" in updates
        or "whisper_initial_prompt" in updates
    ):
        try:
            await _routes._reload_engine()
        except AsrUnavailable as exc:
            raise HTTPException(status_code=503, detail=exc.public_message) from exc

    return {"status": "updated", "changes": list(updates.keys())}


# Keys whose values must not be returned verbatim in API responses.
_REDACTED_INFERENCE_KEYS = frozenset({"api_key"})
_REDACTED_MODEL_CONFIG_KEYS = frozenset({"hf_token"})
_REDACTED_PLACEHOLDER = "*" * 12


def _redact_inference_config(cfg: dict) -> dict:
    """Return *cfg* with secret values replaced by a placeholder."""
    out = dict(cfg)
    for key in _REDACTED_INFERENCE_KEYS:
        if out.get(key):
            out[key] = _REDACTED_PLACEHOLDER
    return out


@router.get("/inference/config")
async def get_inference_config():
    return _redact_inference_config(_routes._load_inference_config())


@router.post("/inference/config")
async def update_inference_config(body: InferenceConfigRequest):
    cfg = body.model_dump()
    if url := cfg.get("base_url"):
        _validate_base_url(url, "base_url")
    _save_inference_config(cfg)
    return {"status": "saved", "config": _redact_inference_config(cfg)}


@router.delete("/inference/config")
async def delete_inference_config():
    if _routes._INFERENCE_CONFIG_FILE.exists():
        _routes._INFERENCE_CONFIG_FILE.unlink()
    if _routes._LEGACY_INFERENCE_CONFIG.exists():
        _routes._LEGACY_INFERENCE_CONFIG.unlink()
    if _routes._LEGACY_PT_INFERENCE_CONFIG.exists():
        _routes._LEGACY_PT_INFERENCE_CONFIG.unlink()
    return {"status": "deleted"}


@router.post("/inference/models")
async def fetch_inference_models(body: InferenceModelsRequest):
    _validate_base_url(body.base_url, "base_url")
    try:
        models = await get_available_models(body.base_url, body.api_key)
        return {"models": models}
    except Exception:
        logger.exception("Failed to fetch inference models from %s", body.base_url)
        raise HTTPException(
            status_code=500, detail="Failed to fetch models from the configured endpoint"
        ) from None


@router.post("/inference/test")
async def test_inference_connection(body: InferenceTestRequest):
    _validate_base_url(body.base_url, "base_url")
    result = await test_inf_conn(body.base_url, body.api_key)
    return result


@router.post("/inference/generate")
async def inference_generate(body: InferenceGenerateRequest):
    cfg = _routes._load_inference_config()
    _validate_base_url(cfg.get("base_url", "http://localhost:11434/v1"), "base_url")
    engine = InferenceEngine(
        base_url=cfg.get("base_url", "http://localhost:11434/v1"),
        api_key=cfg.get("api_key", "not-needed"),
        model_name=cfg.get("model_name", ""),
        vision_enabled=cfg.get("vision_enabled", False),
    )
    if body.stream:

        async def stream_generator():
            try:
                gen = await engine.generate(
                    prompt=body.prompt,
                    image_base64=body.image_base64,
                    stream=True,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                )
                async for chunk in gen:
                    yield chunk
            finally:
                await engine.aclose()

        return StreamingResponse(stream_generator(), media_type="text/event-stream")
    else:
        try:
            res = await engine.generate(
                prompt=body.prompt,
                image_base64=body.image_base64,
                stream=False,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
            return {"response": res}
        finally:
            await engine.aclose()

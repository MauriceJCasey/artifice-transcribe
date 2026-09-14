# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from model_harness.contract import EndpointRejected
from model_harness.endpoint_policy import EndpointPolicy
from model_harness.registry import HardwareTier
from model_harness.resolution import resolve_model
from openai import AsyncOpenAI

from artifice_transcribe._retry import retry
from artifice_transcribe.services.token_redaction import redact_token, strip_url_userinfo

logger = logging.getLogger(__name__)

_inference_endpoint_policy = EndpointPolicy()


@retry(max_attempts=4, base_delay=1.0, label="chat.completions.create")
async def _create_chat_completion(
    client: AsyncOpenAI,
    model_name: str,
    messages: list[dict[str, Any]],
    temperature: float,
    max_tokens: int,
):
    """One OpenAI-compatible chat completion, retried on transient network failures.

    ``chat.completions.create`` is the app's transient-failure-prone network
    call — a connection drop or timeout mid-request should be retried, not
    surfaced as a failed summary/cleanup. The decorator retries on the SDK's
    own ``APIConnectionError`` / ``APITimeoutError``.
    """
    return await client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


async def get_available_models(base_url: str, api_key: str | None = None) -> list[str]:
    """Query {base_url} for available models via model_harness.discovery.

    Raises:
        RuntimeError: If the endpoint is unreachable.
    """
    _inference_endpoint_policy.validate_url(base_url)
    from model_harness.discovery import probe_endpoint

    try:
        result = await probe_endpoint(base_url, policy=_inference_endpoint_policy, timeout_s=10)
        if not result.reachable:
            detail = result.hint or f"Cannot reach {base_url}"
            raise RuntimeError(detail)
        return sorted(result.models)
    except RuntimeError:
        raise
    except Exception as exc:
        logger.error("Failed to fetch models from %s: %s", base_url, exc)
        raise


async def test_connection(base_url: str, api_key: str | None = None) -> dict:
    """Test connectivity to the endpoint and return status with details."""
    _inference_endpoint_policy.validate_url(base_url)
    from model_harness.discovery import probe_endpoint

    try:
        result = await probe_endpoint(base_url, policy=_inference_endpoint_policy, timeout_s=10)
        count = len(result.models)
        if result.reachable:
            return {
                "success": True,
                "message": f"Connected successfully! Found {count} model(s).",
                "model_count": count,
            }
        else:
            err_msg = result.hint or "Server not reachable"
            return {
                "success": False,
                "message": f"Server unreachable: {err_msg}",
                "model_count": 0,
            }
    except Exception as exc:
        safe_detail = strip_url_userinfo(redact_token(str(exc)))
        return {
            "success": False,
            "message": f"Server unreachable: {safe_detail}",
            "model_count": 0,
        }


class InferenceEngine:
    """Unified inference adapter using OpenAI-compatible SDK for BYOM support."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = "not-needed",
        model_name: str = "",
        vision_enabled: bool = False,
    ):
        _inference_endpoint_policy.validate_url(base_url)
        self.base_url = base_url
        self.api_key = api_key if api_key and api_key.strip() else "not-needed"
        self.model_name = model_name
        self.vision_enabled = vision_enabled
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)

    async def aclose(self) -> None:
        """Close the underlying HTTP client's connection pool."""
        await self.client.close()

    async def _resolve_model_name(self) -> str:
        """Pick a model when none is configured, from what the server serves.

        This replaces three stacked defects:

        * ``models.data[0].id`` took whatever the server happened to list
          first — potentially an embedding model, which would return
          confident nonsense rather than an error.
        * Falling back to the literal string ``"default"`` named a model no
          provider serves, guaranteeing a 404 one call later, at which point
          the cause was no longer visible.
        * ``except Exception`` swallowed every failure, including an endpoint
          the policy had rejected.

        The result is cached on the instance by the caller, so the model list
        is fetched once per engine rather than per request.

        Raises:
            RuntimeError: naming the endpoint and what to do.
        """
        try:
            listing = await self.client.models.list()
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            raise RuntimeError(
                f"Cannot list models from {self.base_url}: {exc}. "
                "Check the endpoint is running and reachable."
            ) from exc

        installed = [m.id for m in (listing.data or [])]
        if not installed:
            raise RuntimeError(
                f"No models are available at {self.base_url}. "
                "Install one (e.g. 'ollama pull llama3.2:3b') and retry."
            )

        # Transcribe's optional inference endpoint is used for summaries and
        # cleanup — a chat role. Its Whisper/diarization models are a separate
        # stack entirely and are not resolved here.
        for tier in (HardwareTier.LAPTOP, HardwareTier.DESKTOP, HardwareTier.MAC_UNIFIED):
            resolution = resolve_model(
                role="chat",
                installed=installed,
                app="artifice-transcribe",
                tier=tier,
                configured=None,
            )
            if resolution.model_name is not None:
                return resolution.model_name

        raise RuntimeError(
            f"No suitable text model is installed at {self.base_url}. "
            "Install one (e.g. 'ollama pull llama3.2:3b') and retry."
        )

    async def generate(
        self,
        prompt: str,
        image_base64: str | None = None,
        stream: bool = False,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> str | AsyncGenerator[str, None]:
        """Generate response (text or streaming) with optional vision support."""
        content: str | list[dict[str, Any]] = prompt

        if image_base64 and self.vision_enabled:
            content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                        if not image_base64.startswith("data:")
                        else image_base64
                    },
                },
                {"type": "text", "text": prompt},
            ]

        messages = [{"role": "user", "content": content}]

        if not self.model_name:
            self.model_name = await self._resolve_model_name()

        if stream:
            return self._stream_response(messages, temperature, max_tokens)
        else:
            response = await _create_chat_completion(
                self.client, self.model_name, messages, temperature, max_tokens
            )
            return response.choices[0].message.content or ""

    async def _stream_response(
        self, messages: list[dict[str, Any]], temperature: float, max_tokens: int
    ) -> AsyncGenerator[str, None]:
        stream = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

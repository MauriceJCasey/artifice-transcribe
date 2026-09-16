# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Registry lookups, transitive dependency resolution, and size formatting."""

from __future__ import annotations

import os
from pathlib import Path

from model_harness.registry import ASR_MODELS, AsrModelInfo


def hf_cache_dir() -> Path:
    """Return the Hugging Face cache directory (platform-aware)."""
    import platformdirs

    default = Path(platformdirs.user_cache_dir("huggingface", "huggingface")) / "hub"
    env = os.environ.get("HF_HUB_CACHE", "")
    return Path(env) if env else default


def resolve_transitive(key: str) -> list[AsrModelInfo]:
    """Return the full ordered set of models needed for *key*.

    The first entry is always the requested model itself; any dependencies
    follow in the order they are discovered.  Raises ``KeyError`` if *key*
    is not in :data:`~model_harness.registry.ASR_MODELS`.
    """
    info = ASR_MODELS[key]
    seen: set[str] = {key}
    result: list[AsrModelInfo] = [info]

    # Breadth-first so a direct dependency always appears before its own deps.
    queue: list[str] = list(info.depends_on)
    while queue:
        dep_key = queue.pop(0)
        if dep_key in seen:
            continue
        seen.add(dep_key)
        dep_info = ASR_MODELS[dep_key]
        result.append(dep_info)
        queue.extend(d for d in dep_info.depends_on if d not in seen)

    return result


def total_transitive_size(key: str) -> int:
    """Sum of all model weights for *key* and its dependencies."""
    return sum(info.size_bytes for info in resolve_transitive(key))


def requires_token(key: str) -> bool:
    """``True`` if any model in the transitive set needs an HF token."""
    return any(info.requires_hf_token for info in resolve_transitive(key))


def find_registry_key(info: AsrModelInfo) -> str:
    """Reverse-lookup the registry key for an :class:`AsrModelInfo` instance."""
    for k, v in ASR_MODELS.items():
        if v is info:
            return k
    return info.hf_repo


def human_size(num_bytes: int) -> str:
    """Return a human-readable size string (MB or GB)."""
    if num_bytes >= 1_000_000_000:
        return f"{num_bytes / 1_000_000_000:.2f} GB"
    return f"{num_bytes / 1_000_000:.1f} MB"

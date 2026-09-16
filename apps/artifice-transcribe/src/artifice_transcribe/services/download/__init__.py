# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""ASR model download service — consent, size disclosure, and progress.

This module manages model-weight downloads from Hugging Face with explicit
user consent, real byte-level progress reporting, and accurate transitive-size
disclosure.  It does NOT import ``torch`` at module scope — the lightweight
install (no ``--extra asr``) must be able to serve the model-list and consent
endpoints, and this module is the surface that tells a bare install *what* is
available before it downloads anything.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Consent persistence ──────────────────────────────────────────────────────
#
# These functions are defined HERE (not in a consent.py submodule) because
# tests monkeypatch ``artifice_transcribe.services.download._consent_path`` on
# this package module.  For that patch to reach ``_load_consents`` and
# ``_save_consents`` (which call ``_consent_path`` by name), those call sites
# must resolve the name through THIS module's globals — not a submodule's.


def _consent_path() -> Path:
    """Return the per-user consent file path (``platformdirs``, not CWD)."""
    import platformdirs

    data_dir = Path(platformdirs.user_data_dir("artifice-transcribe", "ArtificeSuite"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "model_consent.json"


def _load_consents() -> dict[str, bool]:
    """Return ``{model_key: True}`` for every consented model."""
    path = _consent_path()
    if not path.exists():
        return {}
    try:
        from secure_io import ensure_restricted

        ensure_restricted(path)
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Could not read consent file at %s — treating as empty", path)
        return {}


def _save_consents(data: dict[str, bool]) -> None:
    """Persist consent decisions with OS-appropriate access controls."""
    from secure_io import write_private_json

    write_private_json(_consent_path(), data)


def is_consented(key: str) -> bool:
    """Return ``True`` if the user has recorded consent for *key*."""
    return bool(_load_consents().get(key, False))


def record_consent(key: str, consented: bool = True) -> None:
    """Set consent for *key* to *consented* and persist."""
    data = _load_consents()
    if consented:
        data[key] = True
    else:
        data.pop(key, None)
    _save_consents(data)


def revoke_consent(key: str) -> None:
    """Remove consent for *key*."""
    record_consent(key, consented=False)


# ── Submodule re-exports ─────────────────────────────────────────────────────
#
# Import order matters in one respect: ``is_consented`` (and the rest of the
# consent block above) must already exist as attributes of this
# partially-initialized package before ``manager.py`` runs, because manager.py
# does ``from . import is_consented``.  The worker helpers
# (``_download_with_progress``, ``_CancelledError``) are referenced by
# manager.py as *module attributes* through a dynamic lookup, so their import
# position relative to manager.py is irrelevant.

from .manager import DownloadManager, get_download_manager  # noqa: E402
from .models import DownloadSet, DownloadState, DownloadStatus  # noqa: E402
from .registry import (  # noqa: E402
    find_registry_key,
    hf_cache_dir,
    human_size,
    requires_token,
    resolve_transitive,
    total_transitive_size,
)
from .worker import _CancelledError, _count_cache_bytes, _download_with_progress  # noqa: E402

__all__ = [
    # Public API — consent persistence.
    "is_consented",
    "record_consent",
    "revoke_consent",
    # Public API — registry / dependency resolution / formatting.
    "find_registry_key",
    "hf_cache_dir",
    "human_size",
    "requires_token",
    "resolve_transitive",
    "total_transitive_size",
    # Public API — download state machine.
    "DownloadState",
    "DownloadStatus",
    "DownloadSet",
    # Public API — download manager.
    "DownloadManager",
    "get_download_manager",
    # Internal names kept at their historical package path so existing tests
    # that reach into them (monkeypatching ``_consent_path``, ``_download_with_progress``,
    # or ``_CancelledError``) keep working unchanged.
    "_CancelledError",
    "_consent_path",
    "_count_cache_bytes",
    "_download_with_progress",
    "_load_consents",
    "_save_consents",
]

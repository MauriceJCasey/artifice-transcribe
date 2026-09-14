# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from platformdirs import user_data_dir
from pydantic_settings import BaseSettings, SettingsConfigDict
from secure_io.migration import migrate_legacy_directory, migrate_legacy_file

logger = logging.getLogger(__name__)

# ── Per-user data directory (uses platformdirs to resolve the canonical path) ──
_USER_DATA_PATH = Path(user_data_dir("artifice-transcribe", "ArtificeSuite", ensure_exists=False))
_DEFAULT_DB_PATH = _USER_DATA_PATH / "transcribe.db"
_DEFAULT_DB_URL = f"sqlite+aiosqlite:///{_DEFAULT_DB_PATH}"
_LEGACY_DB_PATH = Path("./data/transcribe.db").resolve()

_DEFAULT_UPLOAD_PATH = _USER_DATA_PATH / "uploads"
_DEFAULT_OUTPUT_PATH = _USER_DATA_PATH / "outputs"
_LEGACY_UPLOAD_PATH = Path("./uploads").resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    hf_token: str = ""
    database_url: str = _DEFAULT_DB_URL
    upload_dir: str = str(_DEFAULT_UPLOAD_PATH)
    output_dir: str = str(_DEFAULT_OUTPUT_PATH)
    whisper_model: str = "base"
    device: str = "auto"
    max_upload_size: int = 524_288_000  # 500 MB

    # Model selection and configuration
    default_whisper_model: str = "base"
    default_device: str = "auto"
    default_hf_token: str = ""
    # Which ASR engine backs the transcription pipeline. "whisperx" is the
    # original WhisperX+alignment backend (CPU or CUDA); "parakeet" selects
    # NVIDIA's Parakeet TDT 1.1B via NeMo (CUDA-only, English-only). The
    # default preserves existing behavior exactly.
    asr_backend: Literal["whisperx", "parakeet"] = "whisperx"
    # Free-text domain instruction passed to Whisper's ``initial_prompt``
    # decoder-conditioning parameter — a plain-language description of the
    # collection, e.g. "This is a 19th-century archaeological field catalogue
    # in German Kurrentschrift". Whisper prepends this to the decoder's prompt,
    # which measurably improves proper-noun and domain-vocabulary accuracy
    # (place names, personal names, organisational acronyms) — exactly the
    # pain point for oral history. Empty string (default) means Whisper's
    # standard decoding with no extra context. Recorded per-job in the job's
    # options sidecar for methods-section citation. Used by the WhisperX
    # backend only — Parakeet TDT has no prompt-conditioning mechanism (see
    # services/parakeet_engine.py, which accepts but ignores it).
    whisper_initial_prompt: str = ""
    diarization_provider: str = "pyannote"
    # Empty means "let WhisperX pick", which is the safe default: WhisperX
    # tracks the pyannote major version it is pinned against (3.8.6 defaults
    # to pyannote/speaker-diarization-community-1). Naming a model here pins
    # it forever. This previously defaulted to
    # "pyannote/speaker-diarization-3.0", which no longer matches the pinned
    # pyannote 4.x and would force a gated download — harmless only because
    # nothing ever passed the value to WhisperX. See
    # services/transcription.py::_ensure_models.
    diarization_model: str = ""
    enable_alignment_model_cache: bool = True

    def model_post_init(self, __context: object) -> None:
        """Run after field population — handles legacy data migration."""
        self._migrate_legacy_db()
        self._migrate_legacy_uploads()

    def _migrate_legacy_db(self) -> None:
        """Move a legacy ``./data/transcribe.db`` to the platform data directory.

        Only runs when ``database_url`` was *not* overridden by the user
        (i.e. when it still equals the computed default).  A user-set
        ``DATABASE_URL`` is left untouched.
        """
        migrate_legacy_file(
            _LEGACY_DB_PATH,
            _DEFAULT_DB_PATH,
            user_overrode_default=(self.database_url != _DEFAULT_DB_URL),
            logger=logger,
        )

    def _migrate_legacy_uploads(self) -> None:
        """Move files from a legacy ``./uploads/`` directory to the platform
        data directory.

        Only runs when ``upload_dir`` was *not* overridden by the user
        (i.e. when it still equals the computed default).  A user-set
        ``UPLOAD_DIR`` is left untouched.
        """
        migrate_legacy_directory(
            _LEGACY_UPLOAD_PATH,
            _DEFAULT_UPLOAD_PATH,
            user_overrode_default=(self.upload_dir != str(_DEFAULT_UPLOAD_PATH)),
            move_mode="files_only",
            collision_is_silent=False,
            cleanup_empty_legacy=True,
            logger=logger,
        )

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def data_path(self) -> Path:
        return _USER_DATA_PATH


settings = Settings()

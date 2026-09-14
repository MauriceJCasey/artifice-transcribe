# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Centralized logging for the Transcribe pipeline.

Usage:
    from artifice_transcribe._logging import get_logger
    log = get_logger("api")
    log.info("Transcription job %s started", job_id)

Thin wrapper around :class:`shared_ui.logging.AppLogger`. See that module for
the shared rotating-file + console implementation; this file only supplies
transcribe's app-specific configuration (root logger name, environment
variable, filename, and default log directory).
"""

from pathlib import Path

from platformdirs import user_data_dir
from shared_ui.logging import AppLogger

LOG_DIR_ENV = "ARTIFICE_TRANSCRIBE_LOG_DIR"
LOG_FILENAME = "artifice-transcribe.log"

# Defaults to a ``logs`` subdirectory of the app's per-user data directory,
# beside the ``transcribe.db``, ``uploads`` and ``outputs`` the app already
# keeps there — mirroring ``config.py``'s ``_USER_DATA_PATH`` (the base of
# ``settings.data_path``) exactly, so a user has one place to look.
_logger = AppLogger(
    package_name="artifice_transcribe",
    log_dir_env=LOG_DIR_ENV,
    log_filename=LOG_FILENAME,
    default_log_dir=lambda: Path(user_data_dir("artifice-transcribe", "ArtificeSuite")) / "logs",
)

log_dir = _logger.log_dir
log_path = _logger.log_path
setup_logging = _logger.setup_logging
reset = _logger.reset
get_logger = _logger.get_logger

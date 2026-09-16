# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared ``clean_consent`` and ``clean_manager`` fixtures for the download tests."""

import pytest
from artifice_transcribe.services.download import DownloadManager


@pytest.fixture
def clean_consent(monkeypatch, tmp_path):
    """Redirect consent storage to a temp directory."""
    consent_path = tmp_path / "model_consent.json"

    def _fake_consent_path():
        return consent_path

    monkeypatch.setattr(
        "artifice_transcribe.services.download._consent_path",
        _fake_consent_path,
    )
    return consent_path


@pytest.fixture
def clean_manager():
    """Return a fresh DownloadManager with no in-flight state."""
    return DownloadManager()

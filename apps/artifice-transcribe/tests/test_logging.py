# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Logging must survive the frozen Windows build.

The app ships as a windowed executable (``console=False`` in
``artifice-transcribe.spec``), so ``sys.stderr`` goes nowhere. Every diagnostic
the app emitted was therefore invisible to the user: a real failure would be
reported to them only as an opaque error with no surrounding context, and
diagnosing it required reproducing the run with stderr redirected. A log on
disk is what makes the next one self-service.
"""

import logging
import sys
import threading
from logging.handlers import RotatingFileHandler

import pytest
from artifice_transcribe import _logging


@pytest.fixture(autouse=True)
def _isolated_logging(tmp_path, monkeypatch):
    """Run every test against its own throwaway log directory."""
    monkeypatch.setenv(_logging.LOG_DIR_ENV, str(tmp_path / "logs"))
    _logging.reset()
    yield
    _logging.reset()


def _read_log() -> str:
    path = _logging.log_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_records_are_written_to_a_file_on_disk(tmp_path):
    log = _logging.get_logger("api")
    log.info("Starting transcription for %s", "interview.wav")

    assert (tmp_path / "logs" / _logging.LOG_FILENAME).exists()
    assert "Starting transcription for interview.wav" in _read_log()


def test_log_dir_is_overridable_by_environment(tmp_path, monkeypatch):
    target = tmp_path / "elsewhere"
    monkeypatch.setenv(_logging.LOG_DIR_ENV, str(target))
    _logging.reset()

    _logging.get_logger("api").warning("redirected")

    assert (target / _logging.LOG_FILENAME).exists()
    assert "redirected" in (target / _logging.LOG_FILENAME).read_text(encoding="utf-8")


def test_nested_module_record_reaches_the_log_file():
    """A module that only does ``logging.getLogger(__name__)`` still lands in the file.

    ``routes.py``, ``config.py`` and the ``services/*`` modules never touch
    ``_logging`` — they call ``logging.getLogger(__name__)``, which resolves to
    ``artifice_transcribe.api.v1.routes`` and the like. Those loggers carry no
    handlers of their own and ``propagate=True`` by default, so their records
    bubble up to the configured ``artifice_transcribe`` root logger. This
    verifies that hierarchy rather than assuming it carries over from OCR.
    """
    _logging.setup_logging()

    nested = logging.getLogger("artifice_transcribe.api.v1.routes")
    nested.info("nested record from a child module")

    assert "nested record from a child module" in _read_log()


def test_log_file_is_utf8_not_the_windows_locale():
    """The handler must not reintroduce the cp1252 bug it exists to record.

    Transcribe's output is multilingual transcripts and redacted tokens, both
    of which carry non-ASCII characters on a normal run — and a
    UnicodeEncodeError inside a handler is swallowed by logging's own error
    handling.
    """
    log = _logging.get_logger("api")
    log.info("Transkription: über Angeschuldigte — 'quoted' \u2014 \u201cfancy\u201d")

    body = _read_log()
    assert "über Angeschuldigte" in body
    assert "\u201cfancy\u201d" in body


def test_file_handler_rotates_rather_than_growing_without_bound():
    _logging.setup_logging()
    handlers = [
        h
        for h in logging.getLogger("artifice_transcribe").handlers
        if isinstance(h, RotatingFileHandler)
    ]
    assert len(handlers) == 1
    assert handlers[0].maxBytes > 0
    assert handlers[0].backupCount >= 1


def test_unwritable_log_dir_does_not_stop_the_app(monkeypatch, tmp_path):
    """A log that cannot be opened must never be the reason a run fails."""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setenv(_logging.LOG_DIR_ENV, str(blocker / "logs"))
    _logging.reset()

    _logging.get_logger("api").info("still works")  # must not raise

    handlers = logging.getLogger("artifice_transcribe").handlers
    assert not any(isinstance(h, RotatingFileHandler) for h in handlers)


def test_no_stderr_handler_when_stderr_is_none(monkeypatch):
    """``pythonw.exe`` sets both std streams to None before anything runs.

    ``logging.StreamHandler(None)`` then falls back to whatever ``sys.stderr``
    is at emit time, so guard it explicitly rather than relying on that.
    """
    monkeypatch.setattr(sys, "stderr", None)
    _logging.reset()
    _logging.setup_logging()

    handlers = logging.getLogger("artifice_transcribe").handlers
    assert not any(type(h) is logging.StreamHandler for h in handlers)
    # The file handler is the whole point in this configuration.
    assert any(isinstance(h, RotatingFileHandler) for h in handlers)


def test_repeated_setup_does_not_duplicate_handlers():
    _logging.setup_logging()
    first = len(logging.getLogger("artifice_transcribe").handlers)
    _logging.setup_logging()
    assert len(logging.getLogger("artifice_transcribe").handlers) == first


def test_reset_allows_reconfiguration():
    _logging.setup_logging()
    root = logging.getLogger("artifice_transcribe")
    assert root.handlers

    _logging.reset()
    assert root.handlers == []

    # After reset, setup_logging must be able to run again and rebuild handlers.
    _logging.setup_logging()
    assert root.handlers


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_unhandled_thread_exception_reaches_the_log():
    """The failure that started all this died in a background worker thread.

    ``threading``'s default excepthook prints to stderr, which the frozen build
    discards — so a worker that crashed in production would leave no trace.
    """
    _logging.setup_logging()

    def boom():
        raise ValueError("reader thread died")

    thread = threading.Thread(target=boom)
    thread.start()
    thread.join()

    body = _read_log()
    assert "reader thread died" in body
    assert "ValueError" in body

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Windowed launcher for ArtificeTranscribe on Windows.

Mirrors `apps/artifice-ocr/launch_ocr_pipeline_web.pyw`: a `.pyw` has no
console, so anything that goes wrong must be written to a log AND shown in a
dialog, or the process simply vanishes with no explanation.

Why this exists at all, rather than launching through WSL: pywebview needs a
platform backend. On Windows it uses the built-in EdgeWebView2 and gets a real
native window for free. Under WSL it needs GTK (`python3-gi`) or Qt (`qtpy`),
neither of which is installed there, so `open_native_window` returns
`opened=False` and the app falls back to a browser tab. Running natively on
Windows is what makes the desktop window work.

Unlike the OCR launcher, this one does not hunt for a working interpreter.
The Windows clone has a `.venv` that `scripts/windows-update.ps1` keeps in
sync, so the honest failure mode is "run the updater", not "silently switch to
some other Python that might be stale".
"""

import os
import sys
import traceback
from pathlib import Path

# apps/artifice-transcribe/ -> repo root
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent.parent
LOG = Path.home() / ".artifice_transcribe" / "launcher.log"

# Third-party imports the app cannot start without. `webview` is deliberately
# absent: a missing window backend is a graceful degradation to the browser,
# not a startup failure.
REQUIRED = (
    "fastapi",
    "uvicorn",
    "sqlalchemy",
    "aiosqlite",
    "pydantic_settings",
    "jinja2",
    "openai",
    "numpy",
)


def _log(message: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(message.rstrip() + "\n")
    except OSError:
        pass


def _show_error(title: str, message: str) -> None:
    _log(f"{title}: {message}")
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass


def _missing(names: tuple[str, ...]) -> list[str]:
    import importlib.util

    missing = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):
            missing.append(name)
    return missing


def _attach_streams() -> None:
    """Give the process real stdout/stderr, writing into the launcher log.

    Under `pythonw.exe` there is no console, so `sys.stdout` and `sys.stderr`
    are None. uvicorn's default logging config builds a StreamHandler over
    `sys.stdout` and dies with a bare "Unable to configure formatter 'default'"
    — a confusing error whose cause is nowhere in the message. Pointing both
    streams at the log file fixes the crash and, usefully, captures the
    server's own output for diagnosis.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        stream = open(LOG, "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def main() -> int:
    os.chdir(ROOT)
    _attach_streams()

    missing = _missing(REQUIRED)
    if missing:
        _show_error(
            "ArtificeTranscribe - missing dependencies",
            "This Python cannot run ArtificeTranscribe.\n\n"
            f"Interpreter:\n{sys.executable}\n\n"
            f"Missing: {', '.join(missing)}\n\n"
            "Run the updater to install them:\n"
            f"    {ROOT}\\scripts\\windows-update.ps1\n\n"
            f"Log: {LOG}",
        )
        return 1

    if _missing(("artifice_transcribe",)):
        _show_error(
            "ArtificeTranscribe - app not installed",
            "The artifice_transcribe package is not importable from this "
            "interpreter.\n\n"
            f"Interpreter:\n{sys.executable}\n\n"
            "Run the updater:\n"
            f"    {ROOT}\\scripts\\windows-update.ps1\n\n"
            f"Log: {LOG}",
        )
        return 1

    from artifice_transcribe.main import cli

    # cli() parses sys.argv. Pass through any extra arguments (e.g. --port) and
    # deliberately do NOT pass --no-window: the native EdgeWebView2 window is
    # the entire reason for launching on Windows rather than through WSL.
    sys.argv = [str(Path(__file__)), *sys.argv[1:]]
    _log(f"Starting ArtificeTranscribe with {sys.executable}")
    cli()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:
        detail = traceback.format_exc()
        _log(detail)
        last = detail.strip().splitlines()[-1] if detail.strip() else "unknown error"
        _show_error(
            "ArtificeTranscribe failed to start",
            f"{last}\n\nFull details in:\n{LOG}",
        )
        sys.exit(1)

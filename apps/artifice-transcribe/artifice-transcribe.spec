# -*- mode: python ; coding: utf-8 -*-
# SPDX-FileCopyrightText: 2026 Maurice Casey
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# PyInstaller spec for artifice-transcribe — Linux/Windows standalone
# executable, CORE-ONLY (no ASR stack bundled).
#
# onedir (not onefile) was chosen deliberately:
#
#   1. Startup speed — onedir maps files directly from disk instead of
#      extracting a self-contained archive to a temp directory on every launch.
#
#   2. Debuggability — you can inspect the _internal/ tree to see exactly
#      what shipped, which onefile compresses into an opaque .pak.
#
#   3. __file__ safety — onefile extracts to a different temp directory each
#      launch, which breaks __file__-relative paths harder.  We already fixed
#      the two remaining __file__ sites, but onedir is the safer choice for
#      a codebase that carries a fossil record of the pattern.
#
#   4. Distribution model — on macOS this becomes an .app bundle; on Windows
#      it is a folder you zip.  Both are standard.  onefile is primarily for
#      CLI tools with no bundled data, which this is not.
#
# Why CORE-ONLY, and why that is the entire point of this spec:
#
#   Transcribe's optional heavy stack (whisperx, torch, torchaudio, pyannote,
#   transformers) is a separate [asr] / [asr-cuda] extra, deliberately kept
#   out of the core dependencies so the app starts, serves its UI, accepts
#   uploads, runs its database and performs manual transcription with no ASR
#   stack installed at all.  A frozen executable cannot pip-install into its
#   own bundled interpreter, so the ASR stack must be installed by the Hub
#   (which has the machinery) — a core-only freeze is the shipping shape, not
#   a compromise.  See docs/TRANSCRIBE_ALIGNMENT_AUDIT.md §5.
#
#   The ASR stack is therefore listed in `excludes` below.  services/
#   transcription.py imports torch/whisperx/transformers lazily (inside the
#   methods that need them), and routes.py imports pyannote.audio lazily, but
#   PyInstaller's bytecode scan follows those function-level imports and would
#   otherwise pull the multi-gigabyte stack in whenever it is installed on the
#   build machine.  Excluding it guarantees the bundle stays core-only no
#   matter what is installed, and the runtime guards (AsrUnavailable) translate
#   the resulting ImportError into the install hint the UI already shows.
#
# Usage (from the repo root):
#   uv run --with pyinstaller pyinstaller --clean --noconfirm \
#       apps/artifice-transcribe/artifice-transcribe.spec

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

# ---------------------------------------------------------------------------
# Configuration — paths relative to the repo root
# ---------------------------------------------------------------------------
APP_NAME = "artifice-transcribe"
PACKAGE = "artifice_transcribe"

# ---------------------------------------------------------------------------
# Hidden imports — uvicorn, fastapi, and friends load things dynamically
# that PyInstaller's static analysis misses.
# ---------------------------------------------------------------------------
HIDDEN_IMPORTS = [
    # uvicorn
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    # fastapi
    "fastapi",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.staticfiles",
    # starlette
    "starlette",
    "starlette.routing",
    "starlette.middleware",
    # jinja2 — PackageLoader loads templates at runtime
    "jinja2.ext",
    # Other dynamic loaders
    "multipart",
    "multipart.multipart",
    # secure_io — imported from routes.py inside function bodies
    "secure_io",
    # aiosqlite — SQLAlchemy's async sqlite dialect loads its DBAPI module
    # dynamically by string name (sqlalchemy.dialects.sqlite.aiosqlite's
    # import_dbapi()), which PyInstaller's static bytecode scan cannot see.
    # Without this, the frozen binary fails at startup with
    # "ModuleNotFoundError: No module named 'aiosqlite'" the moment
    # db/session.py calls create_async_engine() — verified by actually
    # running the built binary, not just building it.
    "aiosqlite",
    "sqlalchemy.dialects.sqlite.aiosqlite",
    # model_harness — the BYOM onboarding surface; listed explicitly so it
    # ships even though every import in transcribe is otherwise static.
    "model_harness",
    "model_harness.contract",
    "model_harness.discovery",
    "model_harness.endpoint_policy",
    "model_harness.registry",
    "model_harness.resolution",
]

# ---------------------------------------------------------------------------
# Exclusions — the ASR stack is deliberately NOT bundled (see header).  This
# is what makes the freeze core-only: even with torch/whisperx installed on
# the build machine, PyInstaller will not collect them, and at runtime the
# lazy imports fail cleanly into the AsrUnavailable guard.
# ---------------------------------------------------------------------------
EXCLUDES = [
    "torch",
    "torchaudio",
    "torchvision",
    "torchcodec",
    "whisperx",
    "transformers",
    "pyannote.audio",
    "triton",
    # pywebview is deliberately NOT bundled either: main.py's cli() already
    # falls back to opening a browser when the native window is unavailable,
    # and the audit doc does not require a native window for Transcribe.  The
    # browser fallback keeps the bundle smaller and free of a GUI toolkit.
    "webview",
]

# ---------------------------------------------------------------------------
# Data files — collect_data_files() picks up everything declared in
# pyproject.toml [tool.setuptools.package-data] for both artifice_transcribe
# (web/static/** and web/templates/**/*.html) and shared_ui.
# ---------------------------------------------------------------------------
datas = []
datas.extend(collect_data_files("artifice_transcribe"))
datas.extend(collect_data_files("shared_ui"))

# ---------------------------------------------------------------------------
# Analysis
#
# We point PyInstaller at a dedicated freeze-entry script (imports
# main.cli() through normal Python import machinery) rather than at main.py
# directly.  Running main.py as a bare script breaks relative imports
# ("from .web.window import ..." → "attempted relative import with no known
# parent package").  The wrapper uses absolute imports and preserves the
# package context.
#
# The pathex list gives PyInstaller the source directories it needs to find
# artifice_transcribe and its workspace dependencies (shared_ui,
# model_harness, secure_io), mirroring the OCR spec so the two stay
# consistent if cross-app imports are ever added.
# ---------------------------------------------------------------------------
_SPEC_DIR = Path(SPECPATH)
_REPO_ROOT = _SPEC_DIR.parent.parent
_FREEZE_ENTRY = str(_REPO_ROOT / "apps" / APP_NAME / "src" / PACKAGE / "_freeze_entry.py")

a = Analysis(
    [_FREEZE_ENTRY],
    pathex=[
        str(_REPO_ROOT / "apps" / "artifice-ocr" / "src"),
        str(_REPO_ROOT / "apps" / "artifice-graph" / "src"),
        str(_REPO_ROOT / "apps" / "artifice-draft" / "src"),
        str(_REPO_ROOT / "apps" / "artifice-transcribe" / "src"),
        str(_REPO_ROOT / "packages" / "shared-ui"),
        str(_REPO_ROOT / "packages" / "model-harness" / "src"),
        str(_REPO_ROOT / "packages" / "secure-io" / "src"),
    ],
    binaries=[],
    datas=datas,
    hiddenimports=HIDDEN_IMPORTS,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
)

pyz = PYZ(a.pure)

# ---------------------------------------------------------------------------
# onedir EXE
# ---------------------------------------------------------------------------
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX compression is disabled for Windows distributions. Packed Python
    # bootloaders and compressed extension modules are a frequent source of
    # Defender ML false positives, while an uncompressed onedir bundle is
    # easier to inspect and sign.
    upx=False,
    # The webview app reports startup failures through its native dialog. A
    # console-bearing GUI executable is unnecessary and looks like a launcher
    # wrapper to endpoint heuristics.
    console=False,
    # Keep the standard least-privilege Windows manifest explicit. The app
    # stores its data under the user's profile and never needs elevation or
    # UIAccess; requesting either would be both unsafe and suspicious.
    uac_admin=False,
    uac_uiaccess=False,
    icon="../../packages/shared-ui/shared_ui/assets/logos/artifice-transcribe.png",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

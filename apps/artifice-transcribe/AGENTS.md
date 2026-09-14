# AGENTS.md

## Status

FastAPI backend for speech-to-text + speaker diarization. Uses WhisperX (Whisper + PyAnnote). Includes a web UI served at `/`.

## Dev Commands

> **Note:** the paths in this file still use an `app/` prefix from before the
> `0979359` web-layer migration. The real package root is
> `src/artifice_transcribe/`, and the web assets are at
> `src/artifice_transcribe/web/static/`. Read `app/x` as
> `src/artifice_transcribe/x` until this file is rewritten.

```bash
uv sync                        # from the REPO ROOT — installs the root
                               # [dependency-groups] dev. There is no per-app
                               # [dev] extra any more; `pip install -e ".[dev]"`
                               # no longer resolves.
python -m artifice_transcribe.main              # run server on :8000, opens web UI at http://127.0.0.1:8000
start.bat                       # Windows shortcut: starts server + opens browser
python tests/test_api.py        # e2e verification (needs audio arg for full test)
ruff check .                    # lint
ruff format .                   # format
```

## Architecture

- **Entry point:** `app/main.py` → FastAPI app with lifespan, serves web UI at `/`
- **Web UI:** `app/static/` — vanilla HTML/CSS/JS SPA (no build step). Served by FastAPI at `/` and `/static/`
- **Routes:** `app/api/v1/routes.py` — all endpoints + background worker
- **Engine:** `app/services/transcription.py` — WhisperX wrapper (lazy-loads models, frees VRAM after every job). Alignment models cached per-language (not cleared by unload). Alignment model loaded per-language on demand. Supports `custom_vocabulary` as Whisper `initial_prompt`.
- **Exports:** `app/services/exports.py` — SRT/VTT/TXT/JSON/MD/PDF/OHMS/TEI formatters
- **ORM:** `app/db/models.py` — `TranscriptionJob` (with oral history metadata), `TranscriptSegment` (with tags), `SpeakerMapping`, `SegmentEditVersion`
- **Config:** `app/config.py` — pydantic-settings, reads `.env`
- **DB:** SQLite by default (`./data/transcribe.db`), async via `aiosqlite`

## Gotchas

- **HF_TOKEN required** for diarization — set in `.env`, no fallback
- **Model is set via `.env`** (`WHISPER_MODEL`), not per-request. The web UI shows the active model as read-only
- **VRAM management:** engine calls `torch.cuda.empty_cache()` + `gc.collect()` after every job. Model is a module-level singleton; do not import `whisperx` at module level (lazy-loaded in `_ensure_models`). Alignment models are kept cached across jobs (small footprint)
- **Background tasks:** uses FastAPI `BackgroundTasks` (in-process). For production scale, swap to Celery/RQ — the `_run_transcription` function in `routes.py` is the unit of work
- **Audio uploads:** stored at `uploads/{job_id}_{filename}`, cleaned up on `DELETE`
- **DB tables:** auto-created on startup via `Base.metadata.create_all`
- **Speaker mapping:** auto-populated during diarization, PATCH-able via API
- **Transcript editing:** segments are editable via contenteditable in the web UI, PATCH endpoint saves to DB, diff view shows word-level changes
- **Export formats:** JSON, SRT, VTT, TXT, Markdown, PDF (PDF requires `fpdf2`)
- **Multilingual:** Whisper auto-detects language per segment. Alignment model is loaded dynamically per detected language (cached in memory)

## Project Layout

```
app/
├── main.py                 # app factory + lifespan + static file serving
├── config.py               # Settings
├── static/                 # Web UI (no build step)
│   ├── index.html          # SPA shell
│   ├── css/app.css         # Design system, light/dark themes
│   └── js/app.js           # All UI logic + API calls
├── db/
│   ├── models.py           # ORM models
│   └── session.py          # async engine
├── schemas/
│   └── transcription.py    # Pydantic schemas
├── services/
│   ├── transcription.py    # WhisperX engine
│   └── exports.py          # format generators
└── api/v1/
    └── routes.py           # endpoints + worker
start.bat                    # Windows shortcut
```

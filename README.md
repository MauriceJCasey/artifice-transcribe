# Artifice Transcribe

[![safety-tests](https://github.com/MauriceJCasey/artifice-transcribe/actions/workflows/safety-tests.yml/badge.svg)](https://github.com/MauriceJCasey/artifice-transcribe/actions/workflows/safety-tests.yml)
[![REUSE status](https://api.reuse.software/badge/github.com/MauriceJCasey/artifice-transcribe)](https://api.reuse.software/info/github.com/MauriceJCasey/artifice-transcribe)

A local-first, bring-your-own-model (BYOM) speech-to-text and speaker-diarization tool for oral
history and interview transcription, built on WhisperX and pyannote.audio. Runs entirely on your
own machine — no audio ever leaves it.

## ⚠️ Development status

**Active development, early stage.** This is provided as-is for developers and advanced users
comfortable troubleshooting local models. Expect bugs, breaking changes, and rough edges.

## Getting started

```bash
uv sync --extra transcribe   # app + web UI, no ASR backend yet
uv sync --extra asr          # + WhisperX/pyannote (CPU)
uv run artifice-transcribe
```

See `apps/artifice-transcribe/` for the application itself and `packages/` for the shared support
libraries it depends on (model harness, output layout, secure I/O, shared UI assets).

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

<!-- BEGIN GENERATED DEPENDENCIES (see scripts/export-to-public-repos.sh) -->
## Dependencies

This list is generated directly from [`apps/artifice-transcribe/pyproject.toml`](apps/artifice-transcribe/pyproject.toml), the actual `dependencies`/`optional-dependencies` tables, not hand-maintained. `.github/workflows/dependency-guard.yml` fails the build if `uv.lock` ever drifts from this file, so any dependency change (including an unreviewed addition) shows up as an explicit, reviewable diff.

**Core:**

- `fastapi>=0.115.0`
- `uvicorn[standard]>=0.30.0`
- `sqlalchemy[asyncio]>=2.0.0`
- `aiosqlite>=0.20.0`
- `pydantic>=2.0.0`
- `pydantic-settings>=2.0.0`
- `python-multipart>=0.0.9`
- `platformdirs>=4.0.0`
- `fpdf2>=2.8.0`
- `openai>=1.0.0`
- `jinja2>=3.0`
- `artifice-model-harness>=0.2.0`
- `numpy>=1.26`
- `artifice-secure-io>=0.2.0`
- `artifice-shared-ui>=0.2.0`
- `artifice-output-layout>=0.1.0`

**`asr` extra:**

- `whisperx>=3.8.0`
- `torch>=2.8.0,<2.9.0`
- `torchaudio>=2.8.0,<2.9.0`
- `torchvision>=0.23.0,<0.24.0`
- `torchcodec>=0.11.0,<0.12.0`
- `pyannote.audio>=4.0`
- `transformers>=4.0`

**`asr-cuda` extra:**

- `whisperx>=3.8.0`
- `torch>=2.8.0,<2.9.0`
- `torchaudio>=2.8.0,<2.9.0`
- `torchvision>=0.23.0,<0.24.0`
- `torchcodec>=0.11.0,<0.12.0`
- `pyannote.audio>=4.0`
- `transformers>=4.0`

**`asr-parakeet` extra:**

- `nemo_toolkit[asr]>=2.7.3`

**`window` extra:**

- `pywebview>=5.0`

For the complete resolved dependency tree (including transitive dependencies), see `uv.lock` at the repo root, or run `uv tree`.
<!-- END GENERATED DEPENDENCIES -->

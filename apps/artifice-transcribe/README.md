# Artifice Transcribe

Local-first, bring-your-own-model speech-to-text and speaker diarization for oral history and
interview transcription. Runs entirely on your own machine — audio never leaves it.

Part of the [Artifice Suite](../../README.md).

## How it works

- **Transcription + diarization** via WhisperX and pyannote.audio, with forced word-level
  alignment (every word locked to an exact audio offset, not just each segment).
- **Optional NVIDIA Parakeet (NeMo) backend** for CUDA hardware — faster, no diarization of its
  own, so it's typically combined with pyannote for speaker labels.
- **Custom vocabulary / initial prompting** to steer recognition toward archival terminology,
  names, and acronyms.
- **Archival metadata** — interviewee/interviewer, date, location, project/collection, access
  restrictions — attached to every job.
- **In-browser editing** — synced audio playback, segment editing with history, and one-click
  speaker relabeling across the whole transcript.
- **Optional AI summarize/cleanup passes** on a finished transcript, via the same
  bring-your-own-model harness the rest of the suite uses (`packages/model-harness`).

## Exports

JSON, SRT, VTT, TXT, Markdown, PDF, OHMS XML, and TEI XML — the last two for digital humanities
and scholarly-edition archives specifically.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). A Hugging Face token is required for
pyannote diarization — accept the user conditions for `pyannote/speaker-diarization-3.1` and
`pyannote/segmentation-3.0` on Hugging Face first, then set `HF_TOKEN`.

```bash
uv sync --extra transcribe        # app + web UI, no ASR backend
uv sync --extra asr               # + WhisperX/pyannote (CPU)
uv sync --extra asr-cuda          # + WhisperX/pyannote (CUDA)
uv sync --extra asr-parakeet      # + NVIDIA Parakeet (CUDA only)

echo 'HF_TOKEN=your_token_here' >> apps/artifice-transcribe/.env
```

The app starts and serves its UI without any ASR extra installed; transcription itself is guarded
by a typed error that names the extra to install if you try it without one.

macOS Apple Silicon: `export PYTORCH_ENABLE_MPS_FALLBACK=1` before launching, for PyTorch MPS
acceleration.

## Usage

```bash
uv run artifice-transcribe
```

- Web UI: `http://127.0.0.1:8000`
- API docs (Swagger): `http://127.0.0.1:8000/docs`

Core endpoints: `POST /api/v1/transcribe` (upload), `GET /api/v1/jobs/{id}` (status),
`GET /api/v1/jobs/{id}/transcript`, `GET /api/v1/jobs/{id}/export?format=...`. The full API
surface (segment editing, speaker remapping, search, summarize/cleanup) is documented at `/docs`.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | required for diarization | Hugging Face token |
| `ARTIFICE_HOST` / `ARTIFICE_PORT` | `127.0.0.1` / `8000` | Bind address |
| `WHISPER_MODEL` | `base` | `tiny` / `base` / `small` / `medium` / `large-v2` / `large-v3` |
| `DEVICE` | `auto` | `cpu` / `cuda` / `auto` |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/transcribe.db` | Async SQLite database path |
| `UPLOAD_DIR` | `./uploads` | Temporary directory for incoming audio |

## Testing

```bash
uv run pytest apps/artifice-transcribe/tests/
```

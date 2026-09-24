# Artifice Transcribe

Local-first transcription for oral history and interviews, with speaker diarization. Built on
[WhisperX](https://github.com/m-bain/whisperX) and
[pyannote.audio](https://github.com/pyannote/pyannote-audio). Audio never leaves your machine.

## What it does

- **Transcribes** recordings with Whisper (or NVIDIA Parakeet, English only, on CUDA), with
  word-level timestamps.
- **Separates speakers** so you can name them once and have the names apply throughout.
- **Reviews** transcripts against synced audio: edit segments, see a diff and edit history,
  and keep a dictionary of names and terms that improves future runs.
- **Records interview metadata**: interviewee, interviewer, date, location, project, access.
- **Exports** to OHMS XML, TEI XML, SRT, VTT, JSON, Markdown, PDF and plain text.
- Optionally **summarises or cleans up** a finished transcript with a text model you connect.

## Run it

From the repository root:

```bash
uv sync --extra transcribe --extra asr   # use --extra asr-cuda for an NVIDIA GPU
uv run artifice-transcribe                # opens http://localhost:8000
```

Models download on first use (roughly 140 MB to 3 GB, depending on size). pyannote's speaker
models are gated on Hugging Face: accept their terms there, then add your token in **Settings**.

## Test

```bash
cd apps/artifice-transcribe && uv run pytest -q
```

## License

AGPL-3.0-or-later.

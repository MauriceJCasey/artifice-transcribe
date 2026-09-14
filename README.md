# Artifice Transcribe

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

<p align="center">
  <img src="packages/shared-ui/shared_ui/assets/logos/artifice-transcribe.png" width="140" alt="Artifice Transcribe logo">
</p>

<h1 align="center">Artifice Transcribe</h1>

<p align="center">
  <a href="https://github.com/MauriceJCasey/artifice-transcribe/actions/workflows/safety-tests.yml"><img src="https://github.com/MauriceJCasey/artifice-transcribe/actions/workflows/safety-tests.yml/badge.svg" alt="safety-tests"></a>
  <a href="https://api.reuse.software/info/github.com/MauriceJCasey/artifice-transcribe"><img src="https://api.reuse.software/badge/github.com/MauriceJCasey/artifice-transcribe" alt="REUSE status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-0.4.0-informational" alt="Version 0.4.0">
</p>

A local-first transcription tool for oral history and interviews, with speaker diarization.

Artifice Transcribe turns an audio or video recording into an editable, speaker-labelled
transcript, using [WhisperX](https://github.com/m-bain/whisperX) for speech recognition and
[pyannote.audio](https://github.com/pyannote/pyannote-audio) to tell speakers apart. You review
it against the audio, name the speakers, add the interview's metadata, and export it in the
formats archives use.

Everything runs on your own machine. Audio never leaves it.

**⚠️ Active development, early stage.** Expect rough edges. This is for people comfortable
troubleshooting local models.

## What it does

- **Transcription.** Whisper, from tiny to large-v3, with word-level timestamps. NVIDIA Parakeet
  is available as an English-only alternative on CUDA.
- **Speakers.** Detects who spoke when. Name a speaker once and the name applies throughout;
  enrol known voices to recognise them in later interviews.
- **Review.** The transcript plays in sync with the audio. Correct any segment in place, compare
  against the original with a diff, and step back through each segment's edit history.
- **Vocabulary.** A persistent dictionary of names, places and terms steers every future run.
- **Metadata.** Interviewee, interviewer, date, location, project, collection ID and access
  restrictions travel with each transcript.
- **Export.** OHMS XML, TEI XML, SRT, VTT, JSON, Markdown, PDF and plain text.
- **Optional text tools.** Connect Ollama, LM Studio or anything OpenAI-compatible to summarise
  or clean up a finished transcript. This never touches transcription itself.

## Getting started

```bash
uv sync --extra transcribe --extra asr   # use --extra asr-cuda for an NVIDIA GPU
uv run artifice-transcribe
```

This opens a local web interface at `http://localhost:8000`. Models download on first use.
pyannote's speaker models are gated on Hugging Face: accept their terms there, then add your
token in **Settings**.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).

<details>
<summary>Dependencies</summary>

<!-- BEGIN GENERATED DEPENDENCIES (see scripts/export-to-public-repos.sh) -->
<!-- END GENERATED DEPENDENCIES -->

</details>

# ArtificeTranscribe

**High-Precision Audio Transcription, Speaker Diarization & Oral History Archiving**

*Part of the [Artifice Suite](../../README.md) — Local-First, Model-Agnostic Software Harnesses for Humanities Research.*

---

## 🏛️ Philosophy: The Software Harness vs. The Chatbot

ArtificeTranscribe is an open-source, production-grade speech-to-text and speaker diarization harness built for oral historians, archivists, researchers, and journalists. It is engineered around Joseph Weizenbaum’s anti-ELIZA principle: **software should perform deterministic audio processing, alignment, and metadata binding, rather than conversational summary loops.**

┌────────────────────────────────────────────────────────────────────────┐
│                     ArtificeTranscribe Harness                         │
│                                                                        │
│   1. Speech Recognition & Diarization (WhisperX + PyAnnote)            │
│   2. Forced Word-Level Alignment (Exact Timestamp Sub-segmentation)    │
│   3. Interactive Remapping (Audio-Synced Speaker & Transcript Editor)  │
│   4. Oral History Archival Export (OHMS XML, TEI XML, PDF, Subtitles)  │
└────────────────────────────────────────────────────────────────────────┘

1. **Deterministic Execution, No Conversational Drift:** ArtificeTranscribe never "chats" about audio recordings. It accepts raw interview media, runs forced word-level alignment, binds archival metadata, and outputs precision-timestamped records.
2. **Forced Word-Level Precision:** Audio transcripts are only as useful as their timing. ArtificeTranscribe locks every individual word to exact audio offsets, eliminating drift and making audio playback perfectly synchronised with the text.
3. **Archival Integrity & Local Privacy:** Confidential oral histories, testimonies, and unpublished field recordings remain 100% offline. Models run on local GPU hardware with strict VRAM hygiene (`torch.cuda.empty_cache()` and `gc.collect()` after every job).
4. **Editorial Visual Identity:** Built using **The New Masses Design System** (`packages/shared-ui`)—a warm, paper-and-ink interface inspired by 1930s radical editorial design and Soviet Constructivism.

---

## ✨ Key Capabilities

### 1. Advanced Speech & Speaker Diarization Engine
- **WhisperX + PyAnnote Integration:** Combines multilingual Whisper transcription with PyAnnote speaker diarization to automatically detect and segment distinct voices (`SPEAKER_00`, `SPEAKER_01`).
- **Forced Word-Level Alignment:** Dynamic language detection per segment with cached alignment models for millisecond-level word timing.
- **Custom Vocabulary & Initial Prompting:** Guides the speech recognition engine toward specialized historical terminology, archival codes, acronyms, and proper nouns.
- **VRAM Lifecycle Hygiene:** Automatic model caching paired with aggressive CUDA cleanup post-job to protect hardware health and prevent memory leaks.

### 2. Archival Oral History Metadata
Attach rich historical provenance metadata directly to transcription jobs:
- **Interviewee & Interviewer** full names
- **Interview Date & Location**
- **Project Name & Collection ID**
- **Access Restrictions & Archival Repository Notes**

### 3. Interactive Web Editing & Audio Sync
- **Synchronised Audio Playback:** Interactive media player with real-time transcript segment highlighting as audio plays.
- **Global Speaker Mapping:** Instantly rename auto-detected speaker tags (e.g., `SPEAKER_00` $\rightarrow$ `Dr. Jane Doe`) across the entire transcript.
- **In-Browser Review:** Inline editing of transcript segments with full edit-history tracking and tag support.

### 4. Comprehensive 8-Format Export Ecosystem
- **OHMS XML:** Oral History Metadata Synchronizer standard for digital humanities archives.
- **TEI XML:** Text Encoding Initiative standard for digital scholarly editions.
- **Subtitles (SRT & VTT):** Standard subtitle files for video production and documentary editing.
- **JSON:** Machine-readable segment data with word-level timestamps and speaker mappings.
- **Markdown (`.md`):** Formatted transcripts ready for static site generators or personal note vaults.
- **PDF:** Printable document exports generated via `fpdf2`.
- **TXT:** Clean plain-text transcripts grouped by speaker.

---

## 🎨 Design System (`packages/shared-ui`)

All visual elements in ArtificeTranscribe adhere to **The New Masses Design System**:
- **Palette:** Warm cream paper (`#f6f3ea`), deep warm black ink (`#1b1813`), Esperanto green accents (`#2f7d45`), and antique gold highlights (`#bf9b30`).
- **Typography:** Playfair Display (Display/Headings), Libre Baskerville (Body/Transcript text), and Archivo (UI Labels/Buttons).
- **Surface Elevation:** Paper-like diffused shadows (`shadow-paper`), audio player waveform integration, and tactile button interactions.

---

## 📂 Monorepo Architecture

ArtificeTranscribe is located at `apps/artifice-transcribe` within the Artifice Suite monorepo and shares core dependencies with partner applications:

```
artifice-suite/
├── apps/
│   └── artifice-transcribe/
│       ├── src/
│       │   ├── main.py             # FastAPI application factory & lifecycle management
│       │   ├── config.py           # Pydantic settings parser (.env integration)
│       │   ├── db/                 # SQLAlchemy async ORM models (Jobs, Segments, Speakers)
│       │   ├── schemas/            # Request/response validation schemas
│       │   ├── services/           # WhisperX engine wrapper & multi-format export drivers
│       │   ├── api/v1/             # REST API routes & background job orchestration
│       │   └── static/             # Responsive SPA (The New Masses UI, audio player sync)
│       ├── tests/                  # Pytest and API verification scripts
│       └── README.md
└── packages/
    ├── shared-ui/                  # The New Masses CSS tokens & web components
    ├── model-harness/             # BYOM connectors (Ollama/LM Studio/PyTorch)
    └── core-types/                # Shared TypeScript & Python data interfaces
```

---

## 🚀 Setup & Prerequisites

### Prerequisites
- **Python 3.11+**
- **PyTorch** with CUDA support (Linux/Windows) or Metal Performance Shaders (MPS) support (macOS Apple Silicon).
- **Hugging Face Token (`HF_TOKEN`):** Required for PyAnnote speaker diarization. Accept user conditions on Hugging Face for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

### macOS Apple Silicon Setup Notes
- To enable Apple Silicon GPU acceleration via PyTorch MPS, set the environment fallback variable before launching:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```
- Alternatively, optional support for Apple's native Metal ML framework (`mlx-whisper`) can be utilized for high-performance Mac speech recognition.

### Installation
From the monorepo root:

```bash
# Install shared packages and app in editable mode
pip install -e packages/core-types -e packages/model-harness -e packages/shared-ui -e apps/artifice-transcribe

# Configure environment variables
cp apps/artifice-transcribe/.env.example apps/artifice-transcribe/.env
```

Set your Hugging Face token inside `apps/artifice-transcribe/.env`:
```env
HF_TOKEN="your_huggingface_access_token_here"
```

---

## 🖥️ Usage & Interfaces

### 1. Launching the Web Server
Start the FastAPI server (launches web interface at `http://127.0.0.1:8000`):
```bash
python -m artifice_transcribe.main
```
- **Web UI**: `http://127.0.0.1:8000`
- **Interactive API Documentation (Swagger UI)**: `http://127.0.0.1:8000/docs`

---

## 🔌 REST API Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| **POST** | `/api/v1/transcribe` | Upload audio file with optional metadata (202 Accepted with `job_id`) |
| **GET** | `/api/v1/jobs/{job_id}` | Check job status, progress percentage, and oral history metadata |
| **GET** | `/api/v1/jobs/{job_id}/transcript` | Retrieve structured timestamped transcript segments & tags |
| **PATCH** | `/api/v1/jobs/{job_id}/transcript` | Edit transcript segment text with change tracking |
| **GET** | `/api/v1/jobs/{job_id}/speakers` | Get speaker label-to-name mappings |
| **PATCH** | `/api/v1/jobs/{job_id}/speakers` | Rename speaker labels (e.g., `SPEAKER_00` $\rightarrow$ `Dr. Jane Doe`) |
| **GET** | `/api/v1/jobs/{job_id}/export` | Export transcript (`format=json|srt|vtt|txt|md|pdf|ohms|tei`) |
| **DELETE** | `/api/v1/jobs/{job_id}` | Delete job, database records, and temporary audio files |
| **GET** | `/health` | Application health check |

---

## ⚙️ Configuration Variables

Configure via `.env` file or environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `HF_TOKEN` | *(required)* | Hugging Face token for PyAnnote diarization models |
| `DATABASE_URL` | `sqlite+aiosqlite:///./data/transcribe.db` | Async SQLite database path |
| `WHISPER_MODEL` | `base` | Model size (`tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`) |
| `DEVICE` | `auto` | Compute device (`cpu`, `cuda`, `auto`) |
| `UPLOAD_DIR` | `./uploads` | Temporary directory for incoming audio files |

---

## 🛠️ Open-Source Extension Points

We welcome contributions from oral historians, digital archivists, and software engineers!

1. **Archival Metadata Exporters (`apps/artifice-transcribe/src/services/exports.py`)**: Implement custom XML or JSON-LD export formatters for regional archival databases.
2. **Audio Preprocessing Drivers**: Add noise suppression or bandpass filtering pre-processing stages for low-quality archival field tapes.
3. **Custom Alignment Models**: Extend forced alignment model maps for low-resource or ancient languages.

---

## 🧪 Testing

Run backend linters and end-to-end API verification tests:
```bash
# Run linters
ruff check apps/artifice-transcribe/

# Run API test suite against an audio file
python apps/artifice-transcribe/tests/test_api.py path/to/sample.wav
```

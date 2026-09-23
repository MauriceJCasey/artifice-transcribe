# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Synthetic, deterministic API payloads for the Transcribe browser baseline.

Every payload here mirrors a real response shape from
``artifice_transcribe.schemas.transcription`` and the ``model_harness`` registry
so the front-end JavaScript can consume them unchanged.  Nothing in this module
touches the database, the ASR stack, the filesystem or the network.
"""

from __future__ import annotations

import struct

# ── Deterministic identifiers ────────────────────────────────────────────────

JOB_ID = "synthetic-job-001"
"""The single completed oral-history job every journey reviews."""

SEGMENT_IDS = ["seg-001", "seg-002", "seg-003"]
"""Three transcript segments, deterministic order."""

SPEAKER_LABELS = ["SPEAKER_00", "SPEAKER_01"]
SPEAKER_NAMES = {"SPEAKER_00": "Interviewer", "SPEAKER_01": "Mrs. Hamilton"}

# The text a test edits into the middle segment.  Deliberately distinct from the
# seeded text so the payload assertion cannot pass by matching the fixture.
EDITED_TEXT = "Well, I remember the launch day very clearly indeed."


def capabilities(asr_available: bool = True) -> dict:
    """``GET /api/v1/capabilities`` — the required-speech readiness signal."""
    if asr_available:
        return {"asr": {"available": True}}
    return {
        "asr": {
            "available": False,
            "reason": "The transcription stack is not installed.",
            "install_hint": "uv sync --extra asr",
        }
    }


def model_config() -> dict:
    """``GET /api/v1/config`` — read-only active model status shown in Settings."""
    return {
        "whisper_model": "base",
        "asr_backend": "whisperx",
        "device": "auto",
        "hf_token": "",
        "diarization_provider": "pyannote",
        "diarization_model": "",
        "enable_alignment_model_cache": True,
        "whisper_initial_prompt": "",
        "available_whisper_models": ["tiny", "base", "small", "medium", "large-v3"],
        "available_asr_backends": ["whisperx", "parakeet"],
        "available_diarization_providers": ["pyannote"],
    }


def health_detailed() -> dict:
    """``GET /api/v1/health/detailed`` — engine shape from
    ``WhisperXEngine.health_check`` with models not yet loaded."""
    return {
        "status": "degraded",
        "engine": {
            "whisper_model": {"name": "base", "state": "untested", "loaded": False},
            "diarization_model": {"state": "untested", "loaded": False},
            "alignment_models": {"state": "untested", "loaded_languages": [], "count": 0},
            "device": "cpu",
            "gpu": None,
            "hf_token_configured": False,
            "last_error": None,
        },
        "database": {"status": "ok"},
    }


def job(status: str = "completed", **overrides) -> dict:
    """``TranscriptionJob`` serialised as ``JobStatusResponse``."""
    payload = {
        "id": JOB_ID,
        "filename": "interview_mrs_hamilton.wav",
        "status": status,
        "progress_percentage": 100.0 if status == "completed" else 0.0,
        "created_at": "2026-09-20T09:30:00Z",
        "completed_at": "2026-09-20T09:31:12Z" if status == "completed" else None,
        "error_message": None,
        "interviewee": "Mrs. Hamilton",
        "interviewer": None,
        "interview_date": "2026-09-18",
        "location": None,
        "project_name": "Belfast shipyard oral history",
        "collection_id": None,
        "access_restrictions": None,
        "custom_vocabulary": None,
    }
    payload.update(overrides)
    return payload


def transcript(segments: list[dict] | None = None) -> dict:
    """``TranscriptResponse`` for the completed job."""
    if segments is None:
        segments = [
            {
                "id": SEGMENT_IDS[0],
                "speaker_label": SPEAKER_LABELS[0],
                "start_time": 0.0,
                "end_time": 3.4,
                "text": "This is the opening question about the shipyard.",
                "tags": [],
            },
            {
                "id": SEGMENT_IDS[1],
                "speaker_label": SPEAKER_LABELS[1],
                "start_time": 3.4,
                "end_time": 8.1,
                "text": "Well, I remember the launch day quite clearly.",
                "tags": ["anecdote"],
            },
            {
                "id": SEGMENT_IDS[2],
                "speaker_label": SPEAKER_LABELS[1],
                "start_time": 8.1,
                "end_time": 12.6,
                "text": "Everyone came down to the docks that morning.",
                "tags": [],
            },
        ]
    return {"job_id": JOB_ID, "segments": segments}


def speakers() -> dict:
    """``SpeakerMapResponse`` — the two detected speakers, already named."""
    return {
        "job_id": JOB_ID,
        "speakers": [
            {"speaker_label": SPEAKER_LABELS[0], "custom_name": SPEAKER_NAMES[SPEAKER_LABELS[0]]},
            {"speaker_label": SPEAKER_LABELS[1], "custom_name": SPEAKER_NAMES[SPEAKER_LABELS[1]]},
        ],
    }


# ── ASR model download fixtures ──────────────────────────────────────────────

ASR_MODEL_KEYS = [
    "whisper-large-v3",
    "parakeet-tdt-1.1b",
    "pyannote-speaker-diarization",
    "pyannote-embedding",
]


def model_summary(key: str) -> dict:
    """``ModelInfoResponse`` for one ASR model key (mirrors model_harness registry)."""
    registry = {
        "whisper-large-v3": {
            "hf_repo": "openai/whisper-large-v3",
            "size_bytes": 3_087_130_976,
            "requires_hf_token": False,
            "description": "OpenAI Whisper large-v3 — multilingual ASR and speech translation",
            "models": None,
        },
        "parakeet-tdt-1.1b": {
            "hf_repo": "nvidia/parakeet-tdt-1.1b",
            "size_bytes": 4_283_136_000,
            "requires_hf_token": False,
            "description": "NVIDIA Parakeet TDT 1.1B — English ASR (NeMo)",
            "models": None,
        },
        "pyannote-speaker-diarization": {
            "hf_repo": "pyannote/speaker-diarization-3.0",
            "size_bytes": 5_905_440,
            "requires_hf_token": True,
            "description": "pyannote speaker diarization 3.0 — identifies who spoke when",
            "models": [
                {
                    "key": "pyannote-speaker-diarization",
                    "hf_repo": "pyannote/speaker-diarization-3.0",
                    "size_bytes": 5_905_440,
                    "size_human": "5.6 MB",
                    "requires_hf_token": True,
                    "description": "pyannote speaker diarization 3.0 — identifies who spoke when",
                },
                {
                    "key": "pyannote-embedding",
                    "hf_repo": "pyannote/embedding",
                    "size_bytes": 96_383_626,
                    "size_human": "91.9 MB",
                    "requires_hf_token": True,
                    "description": (
                        "pyannote speaker embedding — speaker verification and identification"
                    ),
                },
            ],
        },
        "pyannote-embedding": {
            "hf_repo": "pyannote/embedding",
            "size_bytes": 96_383_626,
            "requires_hf_token": True,
            "description": "pyannote speaker embedding — speaker verification and identification",
            "models": None,
        },
    }
    entry = registry[key]
    models = entry.get("models")
    if models is None:
        models = [
            {
                "key": key,
                "hf_repo": entry["hf_repo"],
                "size_bytes": entry["size_bytes"],
                "size_human": _human(entry["size_bytes"]),
                "requires_hf_token": entry["requires_hf_token"],
                "description": entry["description"],
            }
        ]
    total = sum(m["size_bytes"] for m in models)
    return {
        "key": key,
        "models": models,
        "total_size_bytes": total,
        "total_size_human": _human(total),
        "requires_hf_token": entry["requires_hf_token"],
        "cache_directory": "/home/user/.cache/artifice/models",
        "consented": False,
    }


def models_list() -> dict:
    """``GET /api/v1/models`` — every registered ASR model key."""
    return {"models": [model_summary(key) for key in ASR_MODEL_KEYS]}


def download_started(key: str) -> dict:
    """``DownloadStartResponse``."""
    summary = model_summary(key)
    return {
        "key": key,
        "status": "started",
        "model_count": len(summary["models"]),
        "total_size_bytes": summary["total_size_bytes"],
        "total_size_human": summary["total_size_human"],
    }


def consent_response(key: str, consented: bool) -> dict:
    """``ConsentResponse``."""
    return {"key": key, "consented": consented}


# ── BYOM (optional text-model) fixtures ──────────────────────────────────────


def byom_state(configured: bool) -> dict:
    """``GET /api/byom/state`` — optional post-transcription text tools.

    ``recommendations`` is produced by the same registry helper the real
    router uses, so the shape always matches what ``byom.js`` expects.
    """
    from model_harness.byom import byom_recommendations

    return {
        "app": "artifice-transcribe",
        "configured": configured,
        "endpoint": "http://localhost:11434/v1" if configured else None,
        "model": "qwen2.5:7b" if configured else None,
        "roles": ["chat"],
        "recommendations": byom_recommendations("artifice-transcribe"),
    }


def byom_detect() -> dict:
    """``GET /api/byom/detect`` — no local model server reachable."""
    return {"endpoints": []}


# ── Export builder ───────────────────────────────────────────────────────────


def export_txt_body(segments: list[dict], name_map: dict[str, str]) -> str:
    """Build the TXT export the same way ``services.exports.export_txt`` does."""
    lines: list[str] = []
    current_speaker: str | None = None
    for seg in segments:
        name = name_map.get(seg["speaker_label"], seg["speaker_label"])
        if name != current_speaker:
            current_speaker = name
            lines.append(f"\n[{name}]")
        lines.append(seg["text"])
    return "\n".join(lines).strip() + "\n"


# ── Audio fixture ────────────────────────────────────────────────────────────


def silent_wav(seconds: float = 0.5, rate: int = 8000) -> bytes:
    """Return a valid 16-bit mono PCM WAV of silence — offline, synthetic."""
    frames = int(rate * seconds)
    pcm = b"\x00\x00" * frames
    riff = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    fmt = b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    data = b"data" + struct.pack("<I", len(pcm)) + pcm
    return riff + fmt + data


def _human(size_bytes: int) -> str:
    """Rough human-readable size, matching the download service's intent."""
    mb = size_bytes / 1_000_000
    if mb >= 1000:
        return f"{mb / 1000:.1f} GB"
    return f"{mb:.1f} MB"

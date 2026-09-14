# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    queued = "queued"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TranscriptionOptions(BaseModel):
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.queued
    message: str = "Transcription job accepted"


class JobStatusResponse(BaseModel):
    id: str
    filename: str
    status: JobStatus
    progress_percentage: float
    created_at: datetime
    completed_at: datetime | None = None
    error_message: str | None = None
    interviewee: str | None = None
    interviewer: str | None = None
    interview_date: str | None = None
    location: str | None = None
    project_name: str | None = None
    collection_id: str | None = None
    access_restrictions: str | None = None
    custom_vocabulary: str | None = None


class SegmentOut(BaseModel):
    id: str
    speaker_label: str
    start_time: float
    end_time: float
    text: str
    tags: list[str] = []


class TranscriptResponse(BaseModel):
    job_id: str
    segments: list[SegmentOut]


class SpeakerRename(BaseModel):
    speaker_label: str = Field(..., examples=["SPEAKER_00"])
    custom_name: str = Field(..., min_length=1, max_length=128, examples=["Dr. Smith"])


class SpeakerRenameRequest(BaseModel):
    speakers: list[SpeakerRename]


class SpeakerMappingOut(BaseModel):
    speaker_label: str
    custom_name: str


class SpeakerMapResponse(BaseModel):
    job_id: str
    speakers: list[SpeakerMappingOut]


class ExportFormat(str, Enum):
    json = "json"
    srt = "srt"
    vtt = "vtt"
    txt = "txt"
    md = "md"
    pdf = "pdf"
    ohms = "ohms"
    tei = "tei"


class SegmentUpdate(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class SegmentUpdateRequest(BaseModel):
    updates: list[dict] = Field(..., examples=[[{"segment_id": "abc123", "text": "Hello world"}]])


class SegmentUpdateResponse(BaseModel):
    job_id: str
    updated_count: int


class JobMetadataUpdate(BaseModel):
    interviewee: str | None = None
    interviewer: str | None = None
    interview_date: str | None = None
    location: str | None = None
    project_name: str | None = None
    collection_id: str | None = None
    access_restrictions: str | None = None
    custom_vocabulary: str | None = None


class SegmentTagUpdate(BaseModel):
    tags: list[str]


class SearchQuery(BaseModel):
    q: str = Field(..., min_length=1, max_length=200)


class SearchMatch(BaseModel):
    job_id: str
    filename: str
    segment_id: str
    speaker_label: str
    text: str
    start_time: float
    end_time: float
    interviewee: str | None = None
    interviewer: str | None = None
    interview_date: str | None = None
    project_name: str | None = None


class SearchResults(BaseModel):
    results: list[SearchMatch]
    total: int


class EditVersionOut(BaseModel):
    id: str
    segment_id: str
    text_before: str
    text_after: str
    edited_at: datetime


class EditHistoryResponse(BaseModel):
    segment_id: str
    versions: list[EditVersionOut]


class SegmentSplitRequest(BaseModel):
    split_position: int = Field(..., ge=0, description="Character index to split at")


class SegmentSplitResponse(BaseModel):
    segments: list[SegmentOut]


class SegmentMergeResponse(BaseModel):
    segment: SegmentOut
    deleted_segment_id: str


# ── Persistent Dictionary ─────────────────────────────────────────────


class DictionaryResponse(BaseModel):
    id: str
    words: str | None = None
    updated_at: datetime


class DictionaryUpdate(BaseModel):
    words: str | None = None


# ── Speaker Enrollment & Recognition ──────────────────────────────────


class KnownSpeakerOut(BaseModel):
    id: str
    name: str
    model_name: str
    dimension: int
    created_at: datetime
    legacy_embedding: bool = False


class KnownSpeakerList(BaseModel):
    speakers: list[KnownSpeakerOut]


class SpeakerEnrollResponse(BaseModel):
    id: str
    name: str
    message: str = "Speaker enrolled"


class EnrollFromJobRequest(BaseModel):
    job_id: str
    speaker_label: str
    name: str


class SpeakerMatchResult(BaseModel):
    speaker_label: str
    matched_name: str | None = None
    confidence: float | None = None


class SpeakerMatchResponse(BaseModel):
    job_id: str
    matches: list[SpeakerMatchResult]


class SpeakerEmbeddingOut(BaseModel):
    speaker_label: str
    dimension: int
    model_name: str


class InferenceConfigRequest(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"
    model_name: str = ""
    vision_enabled: bool = False


class InferenceTestRequest(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"


class InferenceModelsRequest(BaseModel):
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "not-needed"


class InferenceGenerateRequest(BaseModel):
    prompt: str
    image_base64: str | None = None
    stream: bool = False
    temperature: float = 0.2
    max_tokens: int = 2048


class ModelConfigRequest(BaseModel):
    whisper_model: str | None = None
    asr_backend: str | None = None
    device: str | None = None
    hf_token: str | None = None
    diarization_provider: str | None = None
    diarization_model: str | None = None
    enable_alignment_model_cache: bool | None = None
    whisper_initial_prompt: str | None = None


class ModelConfigResponse(BaseModel):
    whisper_model: str
    asr_backend: str
    device: str
    hf_token: str
    diarization_provider: str
    diarization_model: str
    enable_alignment_model_cache: bool
    whisper_initial_prompt: str
    available_whisper_models: list[str] = Field(
        default_factory=lambda: ["tiny", "base", "small", "medium", "large-v3"]
    )
    available_asr_backends: list[str] = Field(default_factory=lambda: ["whisperx", "parakeet"])
    available_diarization_providers: list[str] = Field(default_factory=lambda: ["pyannote"])


# ── ASR Model Download ─────────────────────────────────────────────────


class ModelDownloadInfo(BaseModel):
    """Information about a single model in a download set."""

    key: str
    hf_repo: str
    size_bytes: int
    size_human: str
    requires_hf_token: bool
    description: str


class ModelInfoResponse(BaseModel):
    """Full download info for a model key — shown in a consent dialog."""

    key: str
    models: list[ModelDownloadInfo]
    total_size_bytes: int
    total_size_human: str
    requires_hf_token: bool
    cache_directory: str
    consented: bool


class ConsentRequest(BaseModel):
    """Consent to download a model — body is explicit consent boolean."""

    consent: bool = Field(..., description="True to grant consent, False to revoke")


class ConsentResponse(BaseModel):
    key: str
    consented: bool


class DownloadStartResponse(BaseModel):
    key: str
    status: str = "started"
    model_count: int
    total_size_bytes: int
    total_size_human: str


class ModelListResponse(BaseModel):
    """Summary of all available ASR models."""

    models: list[ModelInfoResponse]

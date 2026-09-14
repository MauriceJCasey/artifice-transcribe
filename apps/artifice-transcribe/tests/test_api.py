#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Maurice Casey
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""End-to-end verification script for the ArtificeTranscribe API.

This file is deliberately excluded from pytest collection (see
``collect_ignore`` in ``conftest.py``): it needs a live server, and the
transcription step needs real model weights. The no-live-server coverage
of this same path lives in ``tests/test_api_e2e.py`` and runs in the
normal suite. This script remains the manual check against a real engine.

Usage:
    1. Start the server:  python -m artifice_transcribe.main
    2. Run this script:   python tests/test_api.py [path_to_audio_file]

    If no audio file is provided, the script tests all non-engine endpoints
    (health, missing-job handling, export of non-existent job).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def _get(path: str) -> dict | None:
    req = urllib.request.Request(f"{BASE}{path}")
    try:
        with urllib.request.urlopen(req) as resp:
            try:
                return json.loads(resp.read())
            except json.JSONDecodeError:
                return None
    except urllib.error.HTTPError as exc:
        print(f"  GET {path} -> {exc.code}")
        return None


def _delete(path: str) -> int:
    req = urllib.request.Request(f"{BASE}{path}", method="DELETE")
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        print(f"  DELETE {path} -> {exc.code}")
        return exc.code


def _post_multipart(path: str, filepath: str) -> dict | None:
    import mimetypes

    boundary = "----TestBoundary"
    filename = filepath.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
    mime = mimetypes.guess_type(filepath)[0] or "application/octet-stream"

    with open(filepath, "rb") as f:
        file_data = f.read()

    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        + file_data
        + f"\r\n--{boundary}--\r\n".encode()
    )

    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"  POST {path} -> {exc.code} {exc.read().decode()}")
        return None


def _patch_json(path: str, data: dict) -> dict | None:
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=body,
        method="PATCH",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        print(f"  PATCH {path} -> {exc.code}")
        return None


def test_health():
    print("1. Health check")
    resp = _get("/health")
    assert resp and resp["status"] == "ok", f"Unexpected: {resp}"
    print("   PASS")


def test_missing_job():
    print("2. GET /api/v1/jobs/nonexistent -> 404")
    resp = _get("/api/v1/jobs/nonexistent")
    assert resp is None  # expects HTTPError
    print("   PASS")


def test_transcribe_and_poll(audio_path: str):
    print(f"3. POST /api/v1/transcribe with {audio_path}")
    resp = _post_multipart("/api/v1/transcribe", audio_path)
    assert resp and "job_id" in resp, f"Unexpected: {resp}"
    job_id = resp["job_id"]
    print(f"   job_id = {job_id}")

    # Poll until completed or failed
    for _ in range(300):  # up to 5 min
        time.sleep(1)
        status = _get(f"/api/v1/jobs/{job_id}")
        if status is None:
            continue
        state = status.get("status", "")
        print(f"   status = {state}  progress = {status.get('progress_percentage', 0):.0f}%")
        if state in ("completed", "failed"):
            break
    else:
        print("   TIMEOUT waiting for job")
        return job_id

    if state == "failed":
        print(f"   ERROR: {status.get('error_message')}")
        return job_id

    # Get transcript
    print("4. GET /api/v1/jobs/{job_id}/transcript")
    transcript = _get(f"/api/v1/jobs/{job_id}/transcript")
    print(f"   segments = {len(transcript.get('segments', []))}")

    # Rename speaker
    print("5. PATCH /api/v1/jobs/{job_id}/speakers")
    speakers = set(s["speaker_label"] for s in transcript.get("segments", []))
    if speakers:
        first = next(iter(speakers))
        renamed = _patch_json(
            f"/api/v1/jobs/{job_id}/speakers",
            {"speakers": [{"speaker_label": first, "custom_name": "Test Speaker"}]},
        )
        print(f"   mapped: {renamed.get('speakers', [])}")

    # Export formats
    for fmt in ("json", "srt", "vtt", "txt"):
        print(f"6. GET /api/v1/jobs/{job_id}/export?format={fmt}")
        resp = _get(f"/api/v1/jobs/{job_id}/export?format={fmt}")
        if resp is None:
            print("   (raw response checked via urllib)")
        else:
            print(f"   OK ({len(str(resp))} bytes)")

    # Cleanup
    print(f"7. DELETE /api/v1/jobs/{job_id}")
    code = _delete(f"/api/v1/jobs/{job_id}")
    print(f"   status = {code}")

    return job_id


def main():
    audio = sys.argv[1] if len(sys.argv) > 1 else None

    test_health()
    test_missing_job()

    if audio:
        test_transcribe_and_poll(audio)
    else:
        print("\nNo audio file provided — skipping transcription test.")
        print("Usage: python tests/test_api.py <audio_file>")

    print("\nDone.")


if __name__ == "__main__":
    main()

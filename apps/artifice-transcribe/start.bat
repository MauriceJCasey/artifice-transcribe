@echo off
title ArtificeTranscribe
echo Starting ArtificeTranscribe...
start "" http://127.0.0.1:8000
python -m artifice_transcribe.main

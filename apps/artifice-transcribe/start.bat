@echo off
title Artifice Transcribe
echo Starting Artifice Transcribe...
start "" http://127.0.0.1:8000
python -m artifice_transcribe.main

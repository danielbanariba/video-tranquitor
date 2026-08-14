"""Transcriptores de audio: whisper.cpp (subprocess), WhisperX (directo), OpenAI API y ensemble."""

from video_tranquitor.transcribers.ensemble import transcribe_ensemble
from video_tranquitor.transcribers.openai_api import transcribe_openai
from video_tranquitor.transcribers.whispercpp import (
    transcribe_local,
    whisper_result_to_transcriptions,
)
from video_tranquitor.transcribers.whisperx import (
    transcribe_whisperx,
    whisperx_result_to_transcriptions,
)

__all__ = [
    "transcribe_local",
    "whisper_result_to_transcriptions",
    "transcribe_whisperx",
    "whisperx_result_to_transcriptions",
    "transcribe_openai",
    "transcribe_ensemble",
]

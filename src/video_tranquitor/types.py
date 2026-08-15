"""Modelos de datos del pipeline de transcripción."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# Configuración del pipeline
# ---------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    watch_dir: str
    output_dir: str
    transcriber: Literal["local", "openai", "whisperx", "ensemble"]
    whisperx_model: str
    whisper_cpp_path: str
    whisper_model_path: str
    enable_diarization: bool
    enable_analysis: bool
    enable_obsidian: bool
    enable_toon: bool
    obsidian_vault_path: str
    hf_token: str
    openai_api_key: str
    audio_filter: str
    language: str
    transcription_prompt: str
    transcribe_model: str
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    diarization_exclusive: bool = True
    analysis_provider: Literal["codex", "claude"] = "codex"
    analysis_model: str = ""
    analysis_effort: str = ""
    analysis_passes: int = 1
    target_sample_rate: int


# ---------------------------------------------------------------------------
# Resultado de Whisper
# ---------------------------------------------------------------------------


class WhisperWord(BaseModel):
    word: str
    start: float
    end: float


class WhisperSegment(BaseModel):
    text: str
    start: float
    end: float
    words: list[WhisperWord] = []


class WhisperResult(BaseModel):
    segments: list[WhisperSegment]
    language: str


# ---------------------------------------------------------------------------
# Diarización y atribución
# ---------------------------------------------------------------------------


class DiarizationSegment(BaseModel):
    speaker: str
    start: float
    end: float


class AttributedSegment(BaseModel):
    speaker: str | None
    text: str
    start: float
    end: float


# ---------------------------------------------------------------------------
# Resultado de análisis
# ---------------------------------------------------------------------------


class Requirement(BaseModel):
    id: str
    descripcion: str
    prioridad: Literal["alta", "media", "baja"]


class Accionable(BaseModel):
    responsable: str
    tarea: str
    fecha: str | None = None

    @field_validator("fecha", mode="before")
    @classmethod
    def normalize_fecha(cls, v: object) -> str | None:
        """Convierte null/None a None, descarta fechas vacías."""
        if v is None or v == "":
            return None
        return str(v)


class AnalysisResult(BaseModel):
    resumen: str
    requerimientos: list[Requirement]
    accionables: list[Accionable]
    decisiones: list[str]
    diagrama: str = ""


# ---------------------------------------------------------------------------
# Transcripción (formato TOON / Obsidian)
# ---------------------------------------------------------------------------


class Transcription(BaseModel):
    inicio: str
    fin: str
    texto: str


# ---------------------------------------------------------------------------
# Resultado del pipeline
# ---------------------------------------------------------------------------


class PipelineResult(BaseModel):
    input_file: str
    wav_path: str
    transcription: list[AttributedSegment]
    analysis: AnalysisResult | None
    toon_output_path: str | None
    obsidian_output_path: str | None
    duration_ms: float
    audio_duration_sec: float
    stages_run: list[str]
    whisper_result: WhisperResult | None


# ---------------------------------------------------------------------------
# Resultado ensemble
# ---------------------------------------------------------------------------


class EnsembleResult(BaseModel):
    whisper_result: WhisperResult
    arbitrated: list[Transcription]
    turbo: list[Transcription]
    whisperx: list[Transcription]
    arbitration_used: bool

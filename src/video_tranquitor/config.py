"""Carga de configuración del pipeline desde variables de entorno."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from video_tranquitor.types import PipelineConfig

DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe en espanol con puntuacion clara. "
    "Mantiene nombres propios, numeros y siglas tal como se escuchan."
)

# Sin normalización de volumen a propósito: `loudnorm` costaba 102 de los 112
# segundos del preprocess (resamplea internamente a 192 kHz) y no aportaba nada.
# Whisper ya normaliza al calcular el log-mel. Medido sobre 49 min de audio real:
# la transcripción queda 94.74% palabra-por-palabra igual, difiriendo solo en
# puntuación, y la diarización coincide al 97.0% una vez resuelta la permutación
# de etiquetas. El resto de los filtros SÍ aporta: sin ellos aparece daño real
# en el texto (88.61% de coincidencia).
DEFAULT_AUDIO_FILTER = "highpass=f=80, lowpass=f=12000, afftdn=nf=-25"

# community-1 reemplaza a speaker-diarization-3.1 (requiere pyannote.audio >= 4.0).
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

# Unión de los niveles que aceptan ambos CLIs. Codex soporta `minimal` pero no
# `max`; Claude Code soporta `max` pero no `minimal`.
VALID_EFFORT_LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")


def load_config() -> PipelineConfig:
    """Lee el archivo .env y construye un PipelineConfig validado.

    Raises:
        ValueError: Si una variable de entorno requerida no está definida.
    """
    load_dotenv()

    transcriber = os.environ.get("TRANSCRIBER", "local")
    if transcriber not in ("local", "openai", "whisperx", "ensemble"):
        raise ValueError(
            f"TRANSCRIBER='{transcriber}' no es válido. "
            "Valores aceptados: local, openai, whisperx, ensemble"
        )

    enable_diarization = os.environ.get("ENABLE_DIARIZATION", "").lower() == "true"
    enable_analysis = os.environ.get("ENABLE_ANALYSIS", "true").lower() != "false"

    analysis_provider = os.environ.get("ANALYSIS_PROVIDER", "codex").lower()
    if analysis_provider not in ("codex", "claude"):
        raise ValueError(
            f"ANALYSIS_PROVIDER='{analysis_provider}' no es válido. "
            "Valores aceptados: codex, claude"
        )

    # Varias pasadas del análisis se unen al final. El modelo no es determinista:
    # con la misma transcripción de entrada, una corrida captura datos que otra
    # deja pasar. Más pasadas = más cobertura, a costa de tiempo y cuota.
    analysis_passes_raw = os.environ.get("ANALYSIS_PASSES", "1")
    try:
        analysis_passes = int(analysis_passes_raw)
    except ValueError:
        raise ValueError(
            f"ANALYSIS_PASSES='{analysis_passes_raw}' no es un número entero."
        ) from None
    if analysis_passes < 1:
        raise ValueError(
            f"ANALYSIS_PASSES={analysis_passes} no es válido: tiene que ser 1 o más."
        )

    # `minimal` solo existe en Codex; `max` solo en Claude Code.
    analysis_effort = os.environ.get("ANALYSIS_EFFORT", "").lower()
    if analysis_effort and analysis_effort not in VALID_EFFORT_LEVELS:
        raise ValueError(
            f"ANALYSIS_EFFORT='{analysis_effort}' no es válido. "
            f"Valores aceptados: {', '.join(VALID_EFFORT_LEVELS)}"
        )

    # OPENAI_API_KEY solo es obligatoria si TRANSCRIBER=openai.
    # El CLI `codex` (usado por ensemble y análisis) maneja su propia auth.
    if transcriber == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY es requerida cuando TRANSCRIBER=openai. "
            "Configurala en tu archivo .env o cambiá a TRANSCRIBER=local."
        )

    if transcriber in ("local", "ensemble"):
        if not os.environ.get("WHISPER_CPP_PATH"):
            raise ValueError(
                f"WHISPER_CPP_PATH es requerida cuando TRANSCRIBER={transcriber}. "
                "Configurala en tu archivo .env"
            )
        if not os.environ.get("WHISPER_MODEL_PATH"):
            raise ValueError(
                f"WHISPER_MODEL_PATH es requerida cuando TRANSCRIBER={transcriber}. "
                "Configurala en tu archivo .env"
            )

    if enable_diarization and not os.environ.get("HF_TOKEN"):
        raise ValueError(
            "HF_TOKEN es requerida cuando ENABLE_DIARIZATION=true. "
            "Configurala en tu archivo .env"
        )

    target_sample_rate_raw = os.environ.get("TARGET_SAMPLE_RATE")
    target_sample_rate = (
        int(target_sample_rate_raw) if target_sample_rate_raw else 16000
    )

    return PipelineConfig(
        watch_dir=os.environ.get("WATCH_DIR", "./Audios"),
        output_dir=os.environ.get("OUTPUT_DIR", "./output"),
        transcriber=transcriber,  # type: ignore[arg-type]
        whisper_cpp_path=os.environ.get("WHISPER_CPP_PATH", ""),
        whisper_model_path=os.environ.get("WHISPER_MODEL_PATH", ""),
        enable_diarization=enable_diarization,
        enable_analysis=enable_analysis,
        enable_obsidian=os.environ.get("ENABLE_OBSIDIAN", "true").lower() != "false",
        enable_toon=os.environ.get("ENABLE_TOON", "true").lower() != "false",
        obsidian_vault_path=os.environ.get("OBSIDIAN_VAULT_PATH", ""),
        hf_token=os.environ.get("HF_TOKEN", ""),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        audio_filter=os.environ.get("AUDIO_FILTER", DEFAULT_AUDIO_FILTER),
        language=os.environ.get("LANGUAGE", "es"),
        transcription_prompt=os.environ.get(
            "TRANSCRIPTION_PROMPT", DEFAULT_TRANSCRIPTION_PROMPT
        ),
        transcribe_model=os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
        whisperx_model=os.environ.get("WHISPERX_MODEL", "large-v3"),
        diarization_model=os.environ.get("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL),
        diarization_exclusive=(
            os.environ.get("DIARIZATION_EXCLUSIVE", "true").lower() != "false"
        ),
        analysis_provider=analysis_provider,  # type: ignore[arg-type]
        analysis_model=os.environ.get("ANALYSIS_MODEL", ""),
        analysis_effort=analysis_effort,
        analysis_passes=analysis_passes,
        target_sample_rate=target_sample_rate,
    )

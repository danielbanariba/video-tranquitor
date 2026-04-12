"""Carga de configuración del pipeline desde variables de entorno."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from video_tranquitor.types import PipelineConfig

DEFAULT_TRANSCRIPTION_PROMPT = (
    "Transcribe en espanol con puntuacion clara. "
    "Mantiene nombres propios, numeros y siglas tal como se escuchan."
)

DEFAULT_AUDIO_FILTER = (
    "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, loudnorm=I=-16:TP=-1.5:LRA=11"
)


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

    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY es requerida. Configurala en tu archivo .env"
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
        enable_analysis=os.environ.get("ENABLE_ANALYSIS", "true").lower() != "false",
        enable_obsidian=os.environ.get("ENABLE_OBSIDIAN", "true").lower() != "false",
        enable_toon=os.environ.get("ENABLE_TOON", "true").lower() != "false",
        obsidian_vault_path=os.environ.get(
            "OBSIDIAN_VAULT_PATH",
            "/home/banar/Desktop/obsidian/Farinter/07-Reuniones",
        ),
        hf_token=os.environ.get("HF_TOKEN", ""),
        openai_api_key=os.environ["OPENAI_API_KEY"],
        audio_filter=os.environ.get("AUDIO_FILTER", DEFAULT_AUDIO_FILTER),
        language=os.environ.get("LANGUAGE", "es"),
        transcription_prompt=os.environ.get(
            "TRANSCRIPTION_PROMPT", DEFAULT_TRANSCRIPTION_PROMPT
        ),
        transcribe_model=os.environ.get("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-transcribe"),
        whisperx_model=os.environ.get("WHISPERX_MODEL", "large-v3"),
        target_sample_rate=target_sample_rate,
    )

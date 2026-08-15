"""Transcriptor usando WhisperX como librería Python directa (sin subprocess)."""

from __future__ import annotations

import gc
import logging

from video_tranquitor.transcribers.chunking import result_to_transcriptions
from video_tranquitor.types import (
    PipelineConfig,
    Transcription,
    WhisperResult,
    WhisperSegment,
    WhisperWord,
)

logger = logging.getLogger(__name__)

CHUNK_DURATION_SEC = 120  # 2 minutos


def _to_whisper_result(result: dict, language: str) -> WhisperResult:
    """Convierte la salida de whisperx.align() al tipo interno WhisperResult."""
    segments: list[WhisperSegment] = []
    for seg in result.get("segments", []):
        words: list[WhisperWord] = []
        for w in seg.get("words", []):
            if "start" in w and "end" in w:
                words.append(
                    WhisperWord(
                        word=w.get("word", "").strip(),
                        start=float(w["start"]),
                        end=float(w["end"]),
                    )
                )
        segments.append(
            WhisperSegment(
                text=seg.get("text", "").strip(),
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                words=words,
            )
        )
    return WhisperResult(segments=segments, language=language)


def transcribe_whisperx(
    audio_path: str,
    config: PipelineConfig,
    model_size: str = "large-v3",
) -> WhisperResult:
    """Transcribe un archivo de audio usando WhisperX como librería Python.

    Etapas:
    1. Carga del modelo y transcripción con VAD integrado.
    2. Alineación forzada para timestamps a nivel de palabra (wav2vec2).

    La GPU se libera entre etapas para minimizar el uso de VRAM.

    Args:
        audio_path: Ruta al archivo de audio (WAV recomendado).
        config:     Configuración del pipeline (se usa config.language).
        model_size: Tamaño del modelo de Whisper (default "large-v3").

    Returns:
        WhisperResult con segmentos y palabras con timestamps.
    """
    import torch  # importación tardía: solo necesario al transcribir
    import whisperx  # importación tardía: carga pesada

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    batch_size = 16 if device == "cuda" else 4

    print(f"WhisperX — device: {device}, compute: {compute_type}, modelo: {model_size}")

    # Etapa 1: Transcripción con VAD integrado
    print("  [whisperx] Cargando modelo...")
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        language=config.language,
    )

    print("  [whisperx] Cargando audio...")
    audio = whisperx.load_audio(audio_path)

    print("  [whisperx] Transcribiendo (con VAD integrado)...")
    result = model.transcribe(audio, batch_size=batch_size, language=config.language)

    # Liberar VRAM antes de cargar el modelo de alineación
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    # Etapa 2: Alineación forzada para timestamps a nivel de palabra
    print("  [whisperx] Alineando palabras (wav2vec2)...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=config.language,
            device=device,
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )
        del model_a
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        logger.warning(
            "Alineación de palabras falló, usando timestamps de Whisper: %s", exc
        )

    return _to_whisper_result(result, config.language)


def whisperx_result_to_transcriptions(
    result: WhisperResult,
    chunk_seconds: int = CHUNK_DURATION_SEC,
) -> list[Transcription]:
    """Agrupa los segmentos de WhisperResult en chunks de ``chunk_seconds`` segundos.

    Porta la lógica de ``whisperxResultToTranscriptions`` de TypeScript.
    """
    return result_to_transcriptions(result, chunk_seconds)

"""Transcriptor usando whisper.cpp vía subprocess (binario C++)."""

from __future__ import annotations

import json
import logging
import os
import subprocess

from video_tranquitor.preprocessor import format_time
from video_tranquitor.types import (
    PipelineConfig,
    Transcription,
    WhisperResult,
    WhisperSegment,
    WhisperWord,
)

logger = logging.getLogger(__name__)

WHISPER_TIMEOUT_SEC = 30 * 60  # 30 minutos
CHUNK_DURATION_SEC = 120  # 2 minutos

# Techo para no dejar la máquina sin aire cuando el ensemble corre en paralelo.
MAX_WHISPER_THREADS = 8


def _whisper_threads() -> int:
    """Cantidad de hilos para whisper-cli, derivada del hardware.

    El binario usa 4 por defecto, un número que no tiene nada que ver con la
    máquina donde corre. Con GGML_CUDA=ON el trabajo pesado va a la GPU, así
    que subirlo ayuda poco y competir por todos los núcleos perjudica al leg
    de WhisperX cuando el ensemble corre los dos en paralelo.
    """
    return max(4, min(MAX_WHISPER_THREADS, (os.cpu_count() or 4) // 2))


def _parse_whisper_timestamp(ts: str) -> float:
    """Parsea un timestamp de whisper.cpp en formato "HH:MM:SS,mmm" a segundos (float)."""
    time_part, _, ms_part = ts.partition(",")
    h_str, m_str, s_str = time_part.split(":")
    h, m, s = int(h_str), int(m_str), int(s_str)
    ms = int(ms_part) if ms_part else 0
    return h * 3600 + m * 60 + s + ms / 1000.0


def _parse_whisper_json(json_path: str) -> WhisperResult:
    """Convierte la salida JSON de whisper.cpp en WhisperResult."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    segments: list[WhisperSegment] = []
    for seg in data.get("transcription", []):
        start = seg["offsets"]["from"] / 1000.0
        end = seg["offsets"]["to"] / 1000.0

        words: list[WhisperWord] = []
        for tok in seg.get("tokens", []):
            offsets = tok.get("offsets")
            if offsets and offsets.get("from", -1) >= 0 and offsets.get("to", -1) >= 0:
                words.append(
                    WhisperWord(
                        word=tok.get("text", ""),
                        start=offsets["from"] / 1000.0,
                        end=offsets["to"] / 1000.0,
                    )
                )

        segments.append(
            WhisperSegment(
                text=seg.get("text", "").strip(),
                start=start,
                end=end,
                words=words,
            )
        )

    language = data.get("result", {}).get("language", "es")
    return WhisperResult(segments=segments, language=language)


def transcribe_local(audio_path: str, config: PipelineConfig) -> WhisperResult:
    """Transcribe un archivo de audio usando el binario local whisper.cpp.

    Invoca ``whisper-cli`` con ``--output-json-full`` y parsea el JSON resultante.
    El archivo JSON se elimina al finalizar.

    Raises:
        FileNotFoundError: Si el binario de whisper.cpp no existe (E_WHISPER_NOT_FOUND).
        RuntimeError:      Si whisper.cpp supera el tiempo límite o falla.
    """
    binary_path = config.whisper_cpp_path

    if not os.path.exists(binary_path):
        raise FileNotFoundError(
            f"E_WHISPER_NOT_FOUND: Binario de whisper.cpp no encontrado en: {binary_path}. "
            "Compilá whisper.cpp con soporte CUDA y configurá WHISPER_CPP_PATH en tu .env"
        )

    # whisper-cli escribe <audio_path>.json cuando se usa --output-json-full
    output_base = audio_path
    json_output_path = f"{audio_path}.json"

    args = [
        binary_path,
        "-m", config.whisper_model_path,
        "-f", audio_path,
        "-l", config.language,
        "--threads", str(_whisper_threads()),
        "--output-json-full",
        "--output-file", output_base,
        "--no-prints",
    ]

    print("Ejecutando whisper.cpp (GPU/CUDA)...")

    try:
        subprocess.run(
            args,
            timeout=WHISPER_TIMEOUT_SEC,
            check=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"whisper.cpp superó el tiempo límite de 30 minutos para: {audio_path}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else "(sin salida)"
        raise RuntimeError(
            f"Error al ejecutar whisper.cpp: returncode={exc.returncode}\nStderr: {stderr}"
        ) from exc

    if not os.path.exists(json_output_path):
        raise RuntimeError(
            f"whisper.cpp no generó el archivo JSON esperado en: {json_output_path}"
        )

    try:
        return _parse_whisper_json(json_output_path)
    finally:
        if os.path.exists(json_output_path):
            os.unlink(json_output_path)


def whisper_result_to_transcriptions(
    result: WhisperResult,
    chunk_seconds: int = CHUNK_DURATION_SEC,
) -> list[Transcription]:
    """Agrupa los segmentos de WhisperResult en chunks de ``chunk_seconds`` segundos.

    Cada chunk contiene la concatenación de todos los textos de segmentos que
    caen dentro de esa ventana temporal.
    """
    if not result.segments:
        return []

    transcriptions: list[Transcription] = []
    chunk_start = result.segments[0].start
    chunk_end = chunk_start + chunk_seconds
    chunk_texts: list[str] = []

    def _flush(actual_end: float) -> None:
        text = " ".join(chunk_texts).strip()
        if text:
            transcriptions.append(
                Transcription(
                    inicio=format_time(chunk_start),
                    fin=format_time(actual_end),
                    texto=text,
                )
            )

    for seg in result.segments:
        if seg.start >= chunk_end:
            _flush(min(seg.start, chunk_end))
            chunk_start = chunk_end
            # Avanzar límites hasta alcanzar el segmento
            while seg.start >= chunk_start + chunk_seconds:
                chunk_start += chunk_seconds
            chunk_end = chunk_start + chunk_seconds
            chunk_texts = []
        chunk_texts.append(seg.text)

    # Vaciar el último chunk
    last_seg = result.segments[-1]
    _flush(last_seg.end)

    return transcriptions

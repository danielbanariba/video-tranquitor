"""Transcriptor legacy usando la API de OpenAI (gpt-4o-transcribe, whisper-1, etc.)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time

from video_tranquitor.preprocessor import format_time, get_audio_duration, get_file_size_mb
from video_tranquitor.types import Transcription

logger = logging.getLogger(__name__)

_CHUNK_LENGTH_SEC = 120  # 2 minutos


def _transcribe_chunk(
    audio_path: str,
    client: object,  # openai.OpenAI — importación tardía
    transcribe_model: str,
    transcription_prompt: str,
    language: str,
) -> str:
    """Transcribe un chunk de audio con la API de OpenAI. Devuelve el texto o ""."""
    try:
        from openai import OpenAI  # noqa: PLC0415 — importación tardía

        assert isinstance(client, OpenAI)

        file_size = get_file_size_mb(audio_path)
        if file_size > 24:
            logger.warning("Chunk grande: %.2f MB", file_size)

        ext = os.path.splitext(audio_path)[1].lower()
        mime = "audio/wav" if ext == ".wav" else "audio/mpeg"

        with open(audio_path, "rb") as raw:
            file_bytes = raw.read()

        from io import BytesIO  # noqa: PLC0415
        audio_file = BytesIO(file_bytes)
        audio_file.name = os.path.basename(audio_path)

        prompt = transcription_prompt.strip()
        response = client.audio.transcriptions.create(
            model=transcribe_model,
            file=(os.path.basename(audio_path), audio_file, mime),
            language=language,
            response_format="json",
            **({"prompt": prompt} if prompt else {}),
        )
        return response.text.strip()
    except Exception as exc:
        logger.error("Error en transcripción con OpenAI: %s", exc)
        return ""


def transcribe_openai(
    audio_path: str,
    openai_api_key: str,
    transcribe_model: str,
    transcription_prompt: str,
    target_sample_rate: int,
    language: str = "es",
) -> list[Transcription]:
    """Transcribe un archivo largo usando la API de OpenAI dividiendo en chunks de 2 minutos.

    Cada chunk se crea con ffmpeg y se envía por separado a la API.
    Los archivos temporales se eliminan al finalizar cada chunk.

    Args:
        audio_path:           Ruta al archivo de audio WAV de entrada.
        openai_api_key:       Clave de API de OpenAI.
        transcribe_model:     Modelo a usar (ej. "gpt-4o-transcribe", "whisper-1").
        transcription_prompt: Prompt contextual para mejorar la transcripción.
        target_sample_rate:   Sample rate para los chunks WAV generados.
        language:             Código de idioma (default "es").

    Returns:
        Lista de Transcription con inicio/fin/texto por chunk.
    """
    from openai import OpenAI  # noqa: PLC0415 — importación tardía

    client = OpenAI(api_key=openai_api_key)
    duration = get_audio_duration(audio_path)
    duration_ms = duration * 1000
    chunk_length_ms = _CHUNK_LENGTH_SEC * 1000
    transcriptions: list[Transcription] = []
    total_chunks = max(
        1,
        int(duration_ms / chunk_length_ms) + (1 if duration_ms % chunk_length_ms else 0),
    )

    temp_dir = tempfile.mkdtemp(prefix="vt-openai-chunks-")

    try:
        chunk_index = 0
        offset_ms = 0
        while offset_ms < duration_ms:
            chunk_index += 1
            start_sec = offset_ms / 1000.0
            duration_sec = min(_CHUNK_LENGTH_SEC, (duration_ms - offset_ms) / 1000.0)
            chunk_filename = os.path.join(temp_dir, f"chunk_{offset_ms}.wav")

            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-i", audio_path,
                        "-ss", str(start_sec),
                        "-t", str(duration_sec),
                        "-ac", "1",
                        "-ar", str(target_sample_rate),
                        "-sample_fmt", "s16",
                        "-c:a", "pcm_s16le",
                        "-y", chunk_filename,
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                logger.error("Error al crear chunk %d/%d", chunk_index, total_chunks)
                offset_ms += int(chunk_length_ms)
                continue

            file_size = get_file_size_mb(chunk_filename)
            print(
                f"Transcribiendo segmento {chunk_index}/{total_chunks} "
                f"(Tamaño: {file_size:.2f} MB)"
            )

            text = _transcribe_chunk(
                chunk_filename, client, transcribe_model, transcription_prompt, language
            )

            if text:
                end_sec = min((offset_ms + chunk_length_ms) / 1000.0, duration)
                transcriptions.append(
                    Transcription(
                        inicio=format_time(start_sec),
                        fin=format_time(end_sec),
                        texto=text,
                    )
                )

            if os.path.exists(chunk_filename):
                os.unlink(chunk_filename)

            time.sleep(0.5)
            offset_ms += int(chunk_length_ms)
    finally:
        import shutil  # noqa: PLC0415
        shutil.rmtree(temp_dir, ignore_errors=True)

    return transcriptions

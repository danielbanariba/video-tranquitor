"""Preprocesamiento de audio mediante ffmpeg/ffprobe."""

from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


def preprocess_audio(
    input_path: str,
    output_path: str,
    audio_filter: str,
    target_sample_rate: int,
) -> bool:
    """Convierte el audio a WAV mono normalizado usando ffmpeg.

    Intenta primero con los filtros de audio. Si falla y hay filtros
    configurados, reintenta sin ellos.

    Returns:
        True si la conversión fue exitosa, False en caso contrario.
    """
    base_command = [
        "ffmpeg",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", str(target_sample_rate),
        "-sample_fmt", "s16",
    ]
    filter_args = ["-af", audio_filter] if audio_filter else []

    def _run(extra_args: list[str]) -> bool:
        cmd = base_command + extra_args + ["-y", output_path]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0

    try:
        if _run(filter_args):
            return True
        raise subprocess.CalledProcessError(1, "ffmpeg")
    except Exception as error:
        if filter_args:
            logger.error(
                "Error al preparar el audio con filtros, reintentando sin filtros: %s",
                error,
            )
            try:
                if _run([]):
                    return True
            except Exception as fallback_error:
                logger.error("Error al preparar el audio: %s", fallback_error)
                return False
        logger.error("Error al preparar el audio: %s", error)
        return False


def get_audio_duration(path: str) -> float:
    """Devuelve la duración del archivo de audio en segundos usando ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def get_file_size_mb(path: str) -> float:
    """Devuelve el tamaño del archivo en megabytes."""
    return os.path.getsize(path) / (1024 * 1024)


def format_time(seconds: float) -> str:
    """Formatea segundos como HH:MM:SS."""
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

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

    def _run(extra_args: list[str]) -> tuple[bool, str]:
        """Corre ffmpeg y devuelve (éxito, motivo).

        El stderr se devuelve en vez de descartarse: esta etapa tarda minutos, y
        cuando falla el motivo real lo tiene ffmpeg, no el returncode.
        """
        cmd = base_command + extra_args + ["-y", output_path]
        try:
            result = subprocess.run(cmd, capture_output=True)
        except OSError as error:  # ffmpeg ausente o no ejecutable
            return False, str(error)
        if result.returncode == 0:
            return True, ""
        stderr = (result.stderr or b"").decode("utf-8", errors="replace").strip()
        return False, stderr[-800:] or f"ffmpeg terminó con código {result.returncode}"

    ok, motivo = _run(filter_args)
    if ok:
        return True

    if filter_args:
        logger.error(
            "ffmpeg falló con los filtros (%s), reintentando sin ellos. ffmpeg dijo: %s",
            audio_filter,
            motivo,
        )
        ok, motivo = _run([])
        if ok:
            return True

    logger.error("Error al preparar el audio. ffmpeg dijo: %s", motivo)
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
    except Exception as error:
        # Devolver 0.0 en silencio hacía que el pipeline reportara "00:00:00" y
        # siguiera como si nada, escondiendo un ffprobe roto o un WAV corrupto.
        logger.warning(
            "ffprobe no pudo leer la duración de %s (%s). Se reporta 0 y el pipeline sigue.",
            path,
            error,
        )
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

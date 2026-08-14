"""Orquestador del pipeline de transcripción de video/audio."""

from __future__ import annotations

import asyncio
import logging
import os
import time

from video_tranquitor.aligner import align_speakers
from video_tranquitor.analyzer import analyze_transcription
from video_tranquitor.diarizer import diarize
from video_tranquitor.preprocessor import format_time, get_audio_duration, preprocess_audio
from video_tranquitor.transcribers.openai_api import transcribe_openai
from video_tranquitor.transcribers.whispercpp import (
    transcribe_local,
    whisper_result_to_transcriptions,
)
from video_tranquitor.transcribers.whisperx import (
    transcribe_whisperx,
    whisperx_result_to_transcriptions,
)
from video_tranquitor.transcribers.ensemble import transcribe_ensemble
from video_tranquitor.types import (
    AnalysisResult,
    AttributedSegment,
    PipelineConfig,
    PipelineResult,
    Transcription,
    WhisperResult,
)
from video_tranquitor.writers.obsidian_writer import write_obsidian_note
from video_tranquitor.writers.toon_writer import write_toon

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: frozenset[str] = frozenset({".mp4", ".mkv", ".avi", ".mov"})
AUDIO_EXTENSIONS: frozenset[str] = frozenset({".ogg", ".mp3", ".wav", ".m4a", ".flac"})


def _stage_log(label: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"  [{label}] completado en {elapsed:.2f}s")


def _time_string_to_seconds(time_str: str) -> float:
    """Convierte "HH:MM:SS" a segundos."""
    parts = time_str.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return float(h * 3600 + m * 60 + s)


async def run_pipeline(file_path: str, config: PipelineConfig) -> PipelineResult:
    """Ejecuta el pipeline completo de transcripción para un archivo de video o audio.

    Etapas:
    1. Preprocesamiento de audio (ffmpeg).
    2. Transcripción (local / openai / whisperx / ensemble).
    3. Diarización de hablantes (pyannote, opcional).
    4. Análisis con IA (Codex, opcional).
    5. Escritura TOON (opcional).
    6. Nota de Obsidian (opcional).

    Args:
        file_path: Ruta al archivo de entrada (video o audio).
        config:    Configuración del pipeline.

    Returns:
        PipelineResult con todos los resultados intermedios y finales.

    Raises:
        RuntimeError: Si el preprocesamiento falla.
    """
    pipeline_start = time.time()
    stages_run: list[str] = []

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ext = os.path.splitext(file_path)[1].lower()
    is_video = ext in VIDEO_EXTENSIONS

    temp_wav_path = os.path.join(config.output_dir, f"temp_{base_name}.wav")

    os.makedirs(config.output_dir, exist_ok=True)

    print(f"\nIniciando pipeline para: {base_name}")

    try:
        # -------------------------------------------------------------------------
        # Etapa 1: Preprocesamiento de audio
        # -------------------------------------------------------------------------
        stage_start = time.time()
        print(
            "Extrayendo y optimizando audio del video..."
            if is_video
            else "Optimizando audio..."
        )

        preprocess_ok = preprocess_audio(
            file_path,
            temp_wav_path,
            config.audio_filter,
            config.target_sample_rate,
        )

        if not preprocess_ok:
            raise RuntimeError(f"No se pudo preprocesar el archivo: {file_path}")

        stages_run.append("preprocess")
        _stage_log("preprocess", stage_start)

        audio_duration_sec = get_audio_duration(temp_wav_path)
        print(f"Duración total del audio: {format_time(audio_duration_sec)}")

        # -------------------------------------------------------------------------
        # Etapa 2: Transcripción
        # -------------------------------------------------------------------------
        stage_start = time.time()
        print("Transcribiendo audio...")

        raw_transcriptions: list[Transcription]
        whisper_result: WhisperResult | None = None

        if config.transcriber == "openai":
            raw_transcriptions = await asyncio.to_thread(
                transcribe_openai,
                temp_wav_path,
                config.openai_api_key,
                config.transcribe_model,
                config.transcription_prompt,
                config.target_sample_rate,
                config.language,
            )
        elif config.transcriber == "whisperx":
            whisper_result = await asyncio.to_thread(
                transcribe_whisperx, temp_wav_path, config, config.whisperx_model
            )
            raw_transcriptions = whisperx_result_to_transcriptions(whisper_result)
        elif config.transcriber == "ensemble":
            ensemble_result = await transcribe_ensemble(temp_wav_path, config)
            whisper_result = ensemble_result.whisper_result
            raw_transcriptions = ensemble_result.arbitrated
        else:
            # config.transcriber == "local"
            whisper_result = await asyncio.to_thread(
                transcribe_local, temp_wav_path, config
            )
            raw_transcriptions = whisper_result_to_transcriptions(whisper_result)

        stages_run.append("transcribe")
        _stage_log("transcribe", stage_start)

        # Mapear Transcription[] -> AttributedSegment[] (sin hablantes aún)
        transcription: list[AttributedSegment] = [
            AttributedSegment(
                speaker=None,
                text=t.texto,
                start=_time_string_to_seconds(t.inicio),
                end=_time_string_to_seconds(t.fin),
            )
            for t in raw_transcriptions
        ]

        # -------------------------------------------------------------------------
        # Etapa 3: Diarización
        # -------------------------------------------------------------------------
        if config.enable_diarization:
            if (
                whisper_result is None
                or not whisper_result.segments
                or not whisper_result.segments[0].words
            ):
                print(
                    "⚠  Diarización requiere timestamps a nivel de palabra. Omitiendo diarización."
                )
            else:
                stage_start = time.time()
                print("Ejecutando diarización de hablantes...")

                diarization_segments = await asyncio.to_thread(
                    diarize, temp_wav_path, config
                )

                if diarization_segments:
                    transcription = align_speakers(whisper_result, diarization_segments)

                stages_run.append("diarization")
                _stage_log("diarization", stage_start)

        # -------------------------------------------------------------------------
        # Etapa 4: Análisis
        # -------------------------------------------------------------------------
        analysis: AnalysisResult | None = None
        if config.enable_analysis:
            stage_start = time.time()
            print("Analizando transcripción con IA...")
            analysis = await analyze_transcription(transcription, config)
            if analysis:
                stages_run.append("analysis")
                _stage_log("analysis", stage_start)
            else:
                print("  Análisis omitido (falló o no disponible). El pipeline continúa.")

        # -------------------------------------------------------------------------
        # Etapa 5: Escritura TOON
        # -------------------------------------------------------------------------
        toon_output_path: str | None = None
        if config.enable_toon:
            stage_start = time.time()
            toon_path = os.path.join(config.output_dir, f"{base_name}_transcription.toon")
            write_toon(raw_transcriptions, toon_path)
            toon_output_path = toon_path
            stages_run.append("toon")
            _stage_log("toon", stage_start)
            print(f"Archivo TOON guardado en: {toon_path}")

        # -------------------------------------------------------------------------
        # Etapa 6: Obsidian
        # -------------------------------------------------------------------------
        obsidian_output_path: str | None = None
        if config.enable_obsidian:
            stage_start = time.time()
            print("Generando nota en Obsidian...")
            try:
                partial_result = PipelineResult(
                    input_file=file_path,
                    wav_path=temp_wav_path,
                    transcription=transcription,
                    analysis=analysis,
                    toon_output_path=toon_output_path,
                    obsidian_output_path=None,
                    duration_ms=(time.time() - pipeline_start) * 1000,
                    audio_duration_sec=audio_duration_sec,
                    stages_run=list(stages_run),
                    whisper_result=whisper_result,
                )
                obsidian_output_path = str(write_obsidian_note(partial_result, config))
                stages_run.append("obsidian")
                _stage_log("obsidian", stage_start)
                print(f"Nota de Obsidian guardada en: {obsidian_output_path}")
            except Exception as exc:
                message = str(exc)
                if message.startswith("E_VAULT_NOT_FOUND"):
                    print(f"  {message}")
                else:
                    logger.error("Error al escribir nota de Obsidian: %s", message)
                print("  Nota de Obsidian omitida. El pipeline continúa.")

        duration_ms = (time.time() - pipeline_start) * 1000
        print(
            f"\nPipeline finalizado para: {base_name} ({duration_ms / 1000:.2f}s total)"
        )

        return PipelineResult(
            input_file=file_path,
            wav_path=temp_wav_path,
            transcription=transcription,
            analysis=analysis,
            toon_output_path=toon_output_path,
            obsidian_output_path=obsidian_output_path,
            duration_ms=duration_ms,
            audio_duration_sec=audio_duration_sec,
            stages_run=stages_run,
            whisper_result=whisper_result,
        )

    finally:
        # El WAV temporal se borra pase lo que pase. Cuando el ensemble murió
        # por falta de VRAM quedó un archivo de casi 100 MB colgado en output/,
        # y el watcher lo ignora por el prefijo temp_, así que nadie lo limpiaba.
        if os.path.exists(temp_wav_path):
            os.unlink(temp_wav_path)

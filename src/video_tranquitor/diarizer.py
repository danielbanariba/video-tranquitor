"""Diarización de hablantes usando pyannote.audio como librería Python directa."""

from __future__ import annotations

import logging

from video_tranquitor.types import DiarizationSegment, PipelineConfig

logger = logging.getLogger(__name__)


def _load_audio_in_memory(audio_path: str) -> dict | None:
    """Lee un WAV PCM y lo devuelve como {"waveform", "sample_rate"} para pyannote.

    pyannote.audio 4.x decodifica con torchcodec, que solo soporta ffmpeg 4-7. Con
    ffmpeg 8+ instalado la decodificación revienta con `NameError: AudioDecoder`.
    Precargar el audio en memoria evita ese camino por completo.

    Devuelve None si el archivo no es un WAV PCM legible, para que el llamador
    caiga al comportamiento anterior (pasarle la ruta a pyannote).
    """
    import wave  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    # ffmpeg escribe pcm_s16le por defecto; contemplamos igual 8 y 32 bits.
    dtypes = {1: np.uint8, 2: np.int16, 4: np.int32}

    try:
        with wave.open(audio_path, "rb") as wav:
            sample_width = wav.getsampwidth()
            dtype = dtypes.get(sample_width)
            if dtype is None:
                logger.info(
                    "WAV de %d bytes por muestra no soportado en memoria; "
                    "delegando la decodificación a pyannote.",
                    sample_width,
                )
                return None

            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, OSError, EOFError) as error:
        logger.info("No se pudo precargar el audio (%s); usando la ruta directa.", error)
        return None

    samples = np.frombuffer(frames, dtype=dtype)
    if sample_width == 1:
        # PCM de 8 bits es sin signo, centrado en 128.
        waveform = (samples.astype(np.float32) - 128.0) / 128.0
    else:
        waveform = samples.astype(np.float32) / float(2 ** (8 * sample_width - 1))

    # pyannote espera forma (channel, time).
    waveform = waveform.reshape(-1, channels).T

    return {
        "waveform": torch.from_numpy(np.ascontiguousarray(waveform)),
        "sample_rate": sample_rate,
    }


def _load_pipeline(checkpoint: str, hf_token: str):
    """Carga un pipeline de pyannote soportando el rename de `use_auth_token` a `token`.

    pyannote.audio 4.x renombró el parámetro a `token`; 3.x solo acepta `use_auth_token`.
    """
    from pyannote.audio import Pipeline  # noqa: PLC0415

    try:
        return Pipeline.from_pretrained(checkpoint, token=hf_token)
    except TypeError:
        # pyannote.audio 3.x
        return Pipeline.from_pretrained(checkpoint, use_auth_token=hf_token)


def _resolve_annotation(output, exclusive: bool):
    """Devuelve la Annotation a recorrer, tolerando la salida vieja y la nueva.

    - pyannote.audio 4.x / community-1: objeto con `speaker_diarization` y
      `exclusive_speaker_diarization` (esta última sin solapamientos, ideal para
      reconciliar con los timestamps del transcriptor).
    - pyannote.audio 3.x / speaker-diarization-3.1: devuelve la Annotation directa.
    """
    if exclusive:
        exclusive_annotation = getattr(output, "exclusive_speaker_diarization", None)
        if exclusive_annotation is not None:
            return exclusive_annotation
        logger.info(
            "El modelo no expone exclusive_speaker_diarization; "
            "usando la diarización estándar."
        )

    return getattr(output, "speaker_diarization", output)


def diarize(audio_path: str, config: PipelineConfig) -> list[DiarizationSegment]:
    """Ejecuta la diarización de hablantes con pyannote.audio.

    Importa pyannote como librería Python directa (sin subprocess).
    El checkpoint se toma de `config.diarization_model`
    (por defecto `pyannote/speaker-diarization-community-1`).

    Degradación graceful:
    - Si enable_diarization es False, devuelve [].
    - Si falta hf_token, imprime advertencia y devuelve [].

    Args:
        audio_path: Ruta al archivo de audio WAV.
        config:     Configuración del pipeline (usa enable_diarization, hf_token,
                    diarization_model y diarization_exclusive).

    Returns:
        Lista de DiarizationSegment ordenada por tiempo de inicio.
    """
    if not config.enable_diarization:
        return []

    if not config.hf_token:
        print(
            "⚠  Diarización: HF_TOKEN no configurado, omitiendo. "
            "Configurá hf_token en tu .env para habilitar la identificación de hablantes."
        )
        return []

    # Importación tardía: solo cargar el modelo pesado cuando se necesita
    import torch  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(
        "Diarización — modelo: %s, device: %s, exclusiva: %s",
        config.diarization_model,
        device,
        config.diarization_exclusive,
    )

    pipeline = _load_pipeline(config.diarization_model, config.hf_token)
    pipeline.to(device)

    # Precargar en memoria evita el decoder de torchcodec (roto con ffmpeg 8+).
    audio_input = _load_audio_in_memory(audio_path) or audio_path

    annotation = _resolve_annotation(
        pipeline(audio_input),
        config.diarization_exclusive,
    )

    segments: list[DiarizationSegment] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker=speaker,
                start=round(float(turn.start), 3),
                end=round(float(turn.end), 3),
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments

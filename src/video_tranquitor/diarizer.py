"""Diarización de hablantes usando pyannote.audio como librería Python directa."""

from __future__ import annotations

import logging

from video_tranquitor.types import DiarizationSegment, PipelineConfig

logger = logging.getLogger(__name__)


def diarize(audio_path: str, config: PipelineConfig) -> list[DiarizationSegment]:
    """Ejecuta la diarización de hablantes con pyannote/speaker-diarization-3.1.

    Importa pyannote como librería Python directa (sin subprocess).

    Degradación graceful:
    - Si enable_diarization es False, devuelve [].
    - Si falta hf_token, imprime advertencia y devuelve [].

    Args:
        audio_path: Ruta al archivo de audio WAV.
        config:     Configuración del pipeline (usa enable_diarization y hf_token).

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
    from pyannote.audio import Pipeline  # noqa: PLC0415

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Diarización — device: %s", device)

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=config.hf_token,
    )
    pipeline.to(device)

    diarization = pipeline(audio_path)

    segments: list[DiarizationSegment] = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                speaker=speaker,
                start=round(float(turn.start), 3),
                end=round(float(turn.end), 3),
            )
        )

    segments.sort(key=lambda s: s.start)
    return segments

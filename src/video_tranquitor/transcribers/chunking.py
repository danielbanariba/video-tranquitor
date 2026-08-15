"""Agrupación de segmentos en chunks temporales, común a todos los transcriptores."""

from __future__ import annotations

from video_tranquitor.preprocessor import format_time
from video_tranquitor.types import Transcription, WhisperResult

CHUNK_DURATION_SEC = 120  # 2 minutos


def result_to_transcriptions(
    result: WhisperResult,
    chunk_seconds: int = CHUNK_DURATION_SEC,
) -> list[Transcription]:
    """Agrupa los segmentos de un WhisperResult en ventanas de ``chunk_seconds``.

    La grilla se ancla en 0, no en el primer segmento. Antes cada transcriptor
    tenía su propia versión de esta función y se anclaban distinto: whisper.cpp
    en ``segments[0].start`` y WhisperX en 0. Con un audio que arranca en
    silencio —lo normal en una reunión— las ventanas quedaban desplazadas entre
    sí, y el árbitro del ensemble terminaba comparando chunks que cubrían
    segundos distintos.

    Args:
        result:        Resultado del transcriptor, con segmentos ordenados.
        chunk_seconds: Ancho de cada ventana en segundos.

    Returns:
        Lista de Transcription con tiempos en formato HH:MM:SS. Las ventanas sin
        habla no aparecen.
    """
    if not result.segments:
        return []

    transcriptions: list[Transcription] = []
    inicio = 0
    fin = chunk_seconds
    textos: list[str] = []

    for seg in result.segments:
        if seg.start >= fin:
            if textos:
                transcriptions.append(
                    Transcription(
                        inicio=format_time(inicio),
                        fin=format_time(fin),
                        texto=" ".join(textos).strip(),
                    )
                )
            # Saltar directo a la ventana del segmento: las intermedias no
            # tienen habla y no deben generar chunks vacíos.
            inicio = int(seg.start / chunk_seconds) * chunk_seconds
            fin = inicio + chunk_seconds
            textos = []
        textos.append(seg.text.strip())

    if textos:
        transcriptions.append(
            Transcription(
                inicio=format_time(inicio),
                # El último chunk cierra donde termina el habla, no donde
                # terminaría la ventana.
                fin=format_time(min(fin, result.segments[-1].end)),
                texto=" ".join(textos).strip(),
            )
        )

    return transcriptions

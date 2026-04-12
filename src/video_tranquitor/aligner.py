"""Alineación de palabras Whisper con segmentos de diarización."""

from __future__ import annotations

import math

from video_tranquitor.types import AttributedSegment, DiarizationSegment, WhisperResult, WhisperWord


def _find_speaker_index(
    word_start: float,
    speakers: list[DiarizationSegment],
    tolerance_sec: float,
) -> int:
    """Busca el índice del segmento de diarización que mejor coincide con word_start.

    Estrategia:
    1. Búsqueda binaria: encuentra el segmento más a la izquierda cuyo
       end + tolerance >= word_start.
    2. Si hay múltiples candidatos, prefiere el que contiene estrictamente
       word_start.
    3. Si ninguno cae dentro de la ventana de tolerancia, elige el más cercano.
    """
    lo = 0
    hi = len(speakers) - 1
    candidate_idx = -1

    while lo <= hi:
        mid = (lo + hi) >> 1
        seg = speakers[mid]

        if seg.end + tolerance_sec >= word_start:
            candidate_idx = mid
            hi = mid - 1
        else:
            lo = mid + 1

    best_idx = -1
    best_dist = math.inf

    start = 0 if candidate_idx == -1 else candidate_idx
    for i in range(start, len(speakers)):
        seg = speakers[i]
        if seg.start - tolerance_sec > word_start:
            break  # fuera de la ventana

        # Contenencia estricta → retorno inmediato
        if seg.start <= word_start <= seg.end:
            return i

        # Rastrear el más cercano por distancia al intervalo [start, end]
        dist = min(abs(word_start - seg.start), abs(word_start - seg.end))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx != -1:
        return best_idx

    # Fallback: segmento más cercano en toda la lista (caso raro)
    nearest_idx = 0
    nearest_dist = math.inf
    for i, seg in enumerate(speakers):
        dist = min(abs(word_start - seg.start), abs(word_start - seg.end))
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_idx = i
    return nearest_idx


def align_speakers(
    whisper_result: WhisperResult,
    speakers: list[DiarizationSegment],
    tolerance_ms: int = 500,
) -> list[AttributedSegment]:
    """Alinea la salida de Whisper con los segmentos de diarización de pyannote.

    Args:
        whisper_result: Resultado completo de Whisper con tokens por palabra.
        speakers:       Segmentos de diarización ordenados (speaker + rango temporal).
        tolerance_ms:   Milisegundos de expansión del límite de diarización
                        al buscar el speaker correspondiente. Por defecto 500 ms.

    Returns:
        Lista de AttributedSegment — palabras consecutivas del mismo speaker
        se fusionan en un solo párrafo.
    """
    if not whisper_result.segments:
        return []

    # Aplanar todas las palabras de todos los segmentos
    words: list[WhisperWord] = [
        word for seg in whisper_result.segments for word in seg.words
    ]

    # Degradación sin diarización → speaker=None por segmento
    if not speakers:
        return [
            AttributedSegment(
                speaker=None,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            )
            for seg in whisper_result.segments
        ]

    if not words:
        # Existen segmentos pero no hay tokens por palabra
        return [
            AttributedSegment(
                speaker=None,
                text=seg.text.strip(),
                start=seg.start,
                end=seg.end,
            )
            for seg in whisper_result.segments
        ]

    tolerance_sec = tolerance_ms / 1000.0

    # Asignar speaker a cada palabra
    word_speakers: list[str] = [
        speakers[_find_speaker_index(word.start, speakers, tolerance_sec)].speaker
        for word in words
    ]

    # Agrupar palabras consecutivas del mismo speaker en párrafos
    result: list[AttributedSegment] = []
    group_start = 0

    for i in range(1, len(words) + 1):
        speaker_changed = i == len(words) or word_speakers[i] != word_speakers[group_start]

        if speaker_changed:
            group_words = words[group_start:i]
            text = "".join(w.word for w in group_words).strip()
            result.append(
                AttributedSegment(
                    speaker=word_speakers[group_start],
                    text=text,
                    start=group_words[0].start,
                    end=group_words[-1].end,
                )
            )
            group_start = i

    return result

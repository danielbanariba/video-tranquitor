"""Tests para video_tranquitor.transcribers.chunking."""

from __future__ import annotations

from video_tranquitor.transcribers.chunking import result_to_transcriptions
from video_tranquitor.types import WhisperResult, WhisperSegment


def make_result(*tramos: tuple[str, float, float]) -> WhisperResult:
    return WhisperResult(
        segments=[
            WhisperSegment(text=t, start=ini, end=fin, words=[]) for t, ini, fin in tramos
        ],
        language="es",
    )


class TestResultToTranscriptions:
    def test_empty_result_returns_empty_list(self) -> None:
        assert result_to_transcriptions(make_result(), 120) == []

    # La grilla se ancla en 0, no en el primer segmento. Antes whisper.cpp
    # anclaba en segments[0].start y WhisperX en 0, así que con silencio inicial
    # los dos transcriptores del ensemble producían ventanas desplazadas y el
    # árbitro comparaba chunks que cubrían segundos distintos.
    def test_grid_is_anchored_at_zero_despite_leading_silence(self) -> None:
        resultado = result_to_transcriptions(
            make_result(("hola", 7.5, 30.0), ("chau", 130.0, 150.0)), 120
        )

        assert resultado[0].inicio == "00:00:00"
        assert resultado[1].inicio == "00:02:00"

    def test_groups_segments_inside_the_same_window(self) -> None:
        resultado = result_to_transcriptions(
            make_result(("uno", 0.0, 10.0), ("dos", 20.0, 30.0), ("tres", 40.0, 50.0)), 120
        )

        assert len(resultado) == 1
        assert resultado[0].texto == "uno dos tres"

    def test_skips_windows_without_speech(self) -> None:
        # Salta de 0 a 600s: las ventanas intermedias no deben aparecer vacías.
        resultado = result_to_transcriptions(
            make_result(("inicio", 0.0, 10.0), ("mucho después", 600.0, 610.0)), 120
        )

        assert len(resultado) == 2
        assert resultado[1].inicio == "00:10:00"

    def test_last_chunk_ends_at_the_last_segment_not_the_window(self) -> None:
        resultado = result_to_transcriptions(make_result(("corto", 0.0, 12.0)), 120)

        assert resultado[0].fin == "00:00:12"

    def test_trims_whitespace_from_segment_text(self) -> None:
        resultado = result_to_transcriptions(
            make_result(("  hola  ", 0.0, 5.0), (" mundo ", 10.0, 15.0)), 120
        )

        assert resultado[0].texto == "hola mundo"

    # El caso que originó el bug: ambos transcriptores deben coincidir.
    def test_both_transcribers_now_produce_identical_windows(self) -> None:
        from video_tranquitor.transcribers.whispercpp import whisper_result_to_transcriptions
        from video_tranquitor.transcribers.whisperx import whisperx_result_to_transcriptions

        resultado = make_result(
            *[(f"frase {i}", 7.5 + i * 30, 7.5 + i * 30 + 25) for i in range(8)]
        )

        desde_cpp = whisper_result_to_transcriptions(resultado, 120)
        desde_wx = whisperx_result_to_transcriptions(resultado, 120)

        assert [(t.inicio, t.fin) for t in desde_cpp] == [(t.inicio, t.fin) for t in desde_wx]

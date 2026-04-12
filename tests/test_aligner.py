"""Tests para video_tranquitor.aligner — port de aligner.test.ts (12 tests)."""

from __future__ import annotations

import pytest

from video_tranquitor.aligner import align_speakers
from video_tranquitor.types import (
    AttributedSegment,
    DiarizationSegment,
    WhisperResult,
    WhisperSegment,
    WhisperWord,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_word(word: str, start: float, end: float) -> WhisperWord:
    return WhisperWord(word=word, start=start, end=end)


def make_segment(
    text: str, start: float, end: float, words: list[WhisperWord]
) -> WhisperSegment:
    return WhisperSegment(text=text, start=start, end=end, words=words)


def make_whisper_result(segments: list[WhisperSegment]) -> WhisperResult:
    return WhisperResult(segments=segments, language="es")


def make_speaker(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAlignSpeakers:
    # 1. Empty whisperResult → returns []
    def test_returns_empty_when_no_segments(self) -> None:
        result = align_speakers(make_whisper_result([]), [])
        assert result == []

    def test_returns_empty_when_no_segments_with_speakers(self) -> None:
        speakers = [make_speaker("SPEAKER_00", 0, 10)]
        result = align_speakers(make_whisper_result([]), speakers)
        assert result == []

    # 2. Empty speakers array → returns segments with speaker=None
    def test_returns_null_speaker_when_speakers_empty(self) -> None:
        whisper = make_whisper_result([
            make_segment("Hola mundo", 0, 3, [make_word("Hola", 0, 1), make_word("mundo", 1, 3)]),
            make_segment("Como estas", 4, 7, [make_word("Como", 4, 5), make_word("estas", 5, 7)]),
        ])

        result = align_speakers(whisper, [])
        assert len(result) == 2
        assert result[0].speaker is None
        assert result[0].text == "Hola mundo"
        assert result[0].start == 0
        assert result[0].end == 3
        assert result[1].speaker is None
        assert result[1].text == "Como estas"

    # 3. Single speaker, all words aligned → one AttributedSegment
    def test_single_speaker_produces_one_segment(self) -> None:
        whisper = make_whisper_result([
            make_segment(
                "Hola como estas",
                0,
                6,
                [
                    make_word("Hola", 0, 2),
                    make_word(" como", 2, 4),
                    make_word(" estas", 4, 6),
                ],
            )
        ])
        speakers = [make_speaker("SPEAKER_00", 0, 6)]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"
        assert result[0].text == "Hola como estas"
        assert result[0].start == 0
        assert result[0].end == 6

    # 4. Two speakers alternating → multiple AttributedSegments
    def test_splits_words_when_two_speakers_alternate(self) -> None:
        whisper = make_whisper_result([
            make_segment(
                "Buenos dias hola",
                0,
                6,
                [
                    make_word("Buenos", 0, 2),
                    make_word(" dias", 2, 4),
                    make_word(" hola", 10, 12),
                ],
            )
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 4),
            make_speaker("SPEAKER_01", 10, 14),
        ]

        result = align_speakers(whisper, speakers)
        assert len(result) == 2
        assert result[0].speaker == "SPEAKER_00"
        assert "Buenos" in result[0].text
        assert result[1].speaker == "SPEAKER_01"
        assert "hola" in result[1].text

    # 5. Word falls in gap → assigned to nearest speaker by temporal distance
    def test_assigns_gap_word_to_nearest_speaker(self) -> None:
        # Gap: speaker 0 ends at 3.0, speaker 1 starts at 7.0
        # Word at 3.5 is 0.5s from speaker 0 end, 3.5s from speaker 1 start → speaker 0
        whisper = make_whisper_result([
            make_segment("Prueba", 3.5, 4.0, [make_word("Prueba", 3.5, 4.0)])
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 3.0),
            make_speaker("SPEAKER_01", 7.0, 10.0),
        ]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"

    def test_assigns_gap_word_to_closer_second_speaker(self) -> None:
        # Gap: speaker 0 ends at 3.0, speaker 1 starts at 7.0
        # Word at 6.6 is 3.6s from speaker 0 end, 0.4s from speaker 1 start → speaker 1
        whisper = make_whisper_result([
            make_segment("Oye", 6.6, 7.0, [make_word("Oye", 6.6, 7.0)])
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 3.0),
            make_speaker("SPEAKER_01", 7.0, 10.0),
        ]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_01"

    # 6. 500ms boundary tolerance
    def test_matches_word_300ms_beyond_speaker_end(self) -> None:
        # speaker 0: 0..5. Word at 5.3 (300ms past end). With 500ms tolerance → within window
        whisper = make_whisper_result([
            make_segment("Listo", 5.3, 5.8, [make_word("Listo", 5.3, 5.8)])
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 5.0),
            make_speaker("SPEAKER_01", 20.0, 25.0),
        ]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"

    def test_600ms_beyond_end_still_nearest_speaker(self) -> None:
        # speaker 0: 0..5. Word at 5.6. 500ms tolerance → speaker0 window is 0..5.5 (missed).
        # Nearest by distance: speaker0 is 0.6s away, speaker1 is 14.4s away → speaker0 (fallback)
        whisper = make_whisper_result([
            make_segment("Bueno", 5.6, 6.0, [make_word("Bueno", 5.6, 6.0)])
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 5.0),
            make_speaker("SPEAKER_01", 20.0, 25.0),
        ]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"

    # 7. Multiple consecutive words from same speaker → grouped into single paragraph
    def test_groups_consecutive_same_speaker_words(self) -> None:
        whisper = make_whisper_result([
            make_segment(
                "La reunion empieza ahora mismo gracias",
                0,
                10,
                [
                    make_word("La", 0, 1),
                    make_word(" reunion", 1, 2),
                    make_word(" empieza", 2, 4),
                    make_word(" ahora", 4, 6),
                    make_word(" mismo", 6, 8),
                    make_word(" gracias", 8, 10),
                ],
            )
        ])
        speakers = [make_speaker("SPEAKER_00", 0, 10)]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker == "SPEAKER_00"
        assert result[0].text == "La reunion empieza ahora mismo gracias"
        assert result[0].start == 0
        assert result[0].end == 10

    def test_aggregates_text_across_multiple_segments_same_speaker(self) -> None:
        whisper = make_whisper_result([
            make_segment("Hola", 0, 2, [make_word("Hola", 0, 2)]),
            make_segment(" mundo", 2, 4, [make_word(" mundo", 2, 4)]),
        ])
        speakers = [make_speaker("SPEAKER_00", 0, 5)]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].text == "Hola mundo"

    # Edge case: segments with no word-level tokens
    def test_degrades_gracefully_when_no_word_tokens(self) -> None:
        whisper = make_whisper_result([
            make_segment("Texto sin palabras", 0, 5, [])
        ])
        speakers = [make_speaker("SPEAKER_00", 0, 5)]

        result = align_speakers(whisper, speakers)
        assert len(result) == 1
        assert result[0].speaker is None
        assert result[0].text == "Texto sin palabras"

    # Custom tolerance
    def test_respects_custom_tolerance_ms(self) -> None:
        # With 0ms tolerance, word 100ms outside boundary won't be in window
        # but is still assigned to nearest speaker by fallback
        whisper = make_whisper_result([
            make_segment("Test", 5.1, 5.5, [make_word("Test", 5.1, 5.5)])
        ])
        speakers = [
            make_speaker("SPEAKER_00", 0, 5.0),
            make_speaker("SPEAKER_01", 20.0, 25.0),
        ]

        result = align_speakers(whisper, speakers, 0)
        assert len(result) == 1
        # SPEAKER_00 is 0.1s away; SPEAKER_01 is 14.9s away
        assert result[0].speaker == "SPEAKER_00"

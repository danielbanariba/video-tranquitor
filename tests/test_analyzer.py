"""Tests para video_tranquitor.analyzer — validación de resultados y armado del prompt."""

from __future__ import annotations

from video_tranquitor.analyzer import (
    _build_prompt,
    _build_transcription_text,
    _validate_analysis_result,
)
from video_tranquitor.types import AttributedSegment


def base_parsed(**overrides: object) -> dict[str, object]:
    """Construye un dict válido tal como lo devolvería Codex, con overrides opcionales."""
    parsed: dict[str, object] = {
        "resumen": "Resumen ejecutivo de la reunion.",
        "requerimientos": [
            {"id": "REQ-001", "descripcion": "Login con OAuth", "prioridad": "alta"}
        ],
        "accionables": [{"responsable": "Ana", "tarea": "Configurar CI/CD", "fecha": "2026-04-15"}],
        "decisiones": ["Migrar a PostgreSQL"],
    }
    parsed.update(overrides)
    return parsed


class TestValidateAnalysisResultDiagrama:
    def test_passes_diagrama_through_onto_result(self) -> None:
        diagrama = 'flowchart TD\n    A["Problema"] --> B["Solución"]'
        result = _validate_analysis_result(base_parsed(diagrama=diagrama))

        assert result.diagrama == diagrama

    def test_defaults_diagrama_to_empty_string_when_absent(self) -> None:
        result = _validate_analysis_result(base_parsed())

        assert result.diagrama == ""

    def test_coerces_diagrama_to_empty_string_when_wrong_type(self) -> None:
        result = _validate_analysis_result(base_parsed(diagrama=123))

        assert result.diagrama == ""


def make_segment(
    text: str, start: float, end: float, speaker: str | None = None
) -> AttributedSegment:
    return AttributedSegment(speaker=speaker, text=text, start=start, end=end)


class TestBuildTranscriptionText:
    # El hablante se conserva: es lo que permite llenar `responsable` en los accionables.
    def test_keeps_speaker_prefix(self) -> None:
        segments = [
            make_segment("Necesitamos doble factor.", 0.0, 6.0, "SPEAKER_00"),
            make_segment("Yo me encargo del SMS.", 6.0, 11.0, "SPEAKER_01"),
        ]

        result = _build_transcription_text(segments)

        assert result == (
            "SPEAKER_00: Necesitamos doble factor.\nSPEAKER_01: Yo me encargo del SMS."
        )

    # El timestamp NO se envía: el schema no pide marcas de tiempo y el orden
    # ya está implícito en el orden de las líneas.
    def test_omits_timestamps(self) -> None:
        segments = [make_segment("Hola.", 3725.0, 3730.0, "SPEAKER_02")]

        result = _build_transcription_text(segments)

        assert "01:02:05" not in result
        assert "(" not in result
        assert result == "SPEAKER_02: Hola."

    # Sin diarización no hay prefijo: solo el texto limpio.
    def test_without_speaker_emits_bare_text(self) -> None:
        segments = [
            make_segment("Primera línea.", 0.0, 2.0),
            make_segment("Segunda línea.", 2.0, 4.0),
        ]

        result = _build_transcription_text(segments)

        assert result == "Primera línea.\nSegunda línea."

    def test_empty_transcription_returns_empty_string(self) -> None:
        assert _build_transcription_text([]) == ""


class TestBuildPrompt:
    # Los identificadores técnicos son justamente lo que el modelo pierde entre
    # corridas (se midió: `mongo-dbcrm-hn` apareció en una y no en la siguiente,
    # con transcripción de entrada idéntica).
    def test_instructs_verbatim_technical_identifiers(self) -> None:
        prompt = _build_prompt("SPEAKER_00: la base es mongo-dbcrm-hn en el servidor 202.")

        assert "textual" in prompt.lower()
        for termino in ("host", "base de datos", "colección", "ruta", "comando"):
            assert termino in prompt.lower(), f"falta '{termino}' en las reglas de preservación"

    # Si en la charla se nombra a la persona, el accionable debe llevar el nombre
    # real y no la etiqueta anónima del diarizador.
    def test_instructs_resolving_speaker_labels_to_names(self) -> None:
        prompt = _build_prompt("SPEAKER_02: yo lo hago, dice Luis.")

        assert "SPEAKER_" in prompt
        assert "nombre real" in prompt.lower()

    def test_still_embeds_the_transcription(self) -> None:
        prompt = _build_prompt("SPEAKER_00: hola.")

        assert "SPEAKER_00: hola." in prompt

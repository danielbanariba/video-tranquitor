"""Tests para video_tranquitor.analyzer._validate_analysis_result (campo diagrama)."""

from __future__ import annotations

from video_tranquitor.analyzer import _validate_analysis_result


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

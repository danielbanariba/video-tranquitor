"""Análisis de transcripciones mediante Codex para extraer requerimientos y accionables."""

from __future__ import annotations

from video_tranquitor.codex_client import call_codex_with_schema
from video_tranquitor.types import (
    Accionable,
    AnalysisResult,
    AttributedSegment,
    PipelineConfig,
    Requirement,
)

_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "resumen": {"type": "string"},
        "requerimientos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "descripcion": {"type": "string"},
                    "prioridad": {"type": "string", "enum": ["alta", "media", "baja"]},
                },
                "required": ["id", "descripcion", "prioridad"],
                "additionalProperties": False,
            },
        },
        "accionables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "responsable": {"type": "string"},
                    "tarea": {"type": "string"},
                    "fecha": {"type": ["string", "null"]},
                },
                "required": ["responsable", "tarea", "fecha"],
                "additionalProperties": False,
            },
        },
        "decisiones": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["resumen", "requerimientos", "accionables", "decisiones"],
    "additionalProperties": False,
}


def _format_seconds(seconds: float) -> str:
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_transcription_text(transcription: list[AttributedSegment]) -> str:
    lines: list[str] = []
    for seg in transcription:
        time_label = _format_seconds(seg.start)
        if seg.speaker:
            lines.append(f"{seg.speaker} ({time_label}): {seg.text}")
        else:
            lines.append(f"({time_label}): {seg.text}")
    return "\n".join(lines)


def _build_prompt(transcription_text: str) -> str:
    return (
        "Eres un experto en ingeniería de requerimientos. Analiza la siguiente transcripción "
        "de una reunión y extrae información estructurada.\n\n"
        f"Transcripción:\n{transcription_text}\n\n"
        "Devuelve un JSON válido con esta estructura:\n"
        '- resumen: Resumen ejecutivo de la reunión en 1-2 párrafos\n'
        '- requerimientos: array con id ("REQ-001"), descripcion, prioridad ("alta"|"media"|"baja")\n'
        '- accionables: array con responsable, tarea, fecha opcional (YYYY-MM-DD)\n'
        '- decisiones: array de strings\n\n'
        "Si no hay requerimientos, accionables o decisiones, devuelve arrays vacíos."
    )


def _validate_analysis_result(parsed: object) -> AnalysisResult:
    """Valida y normaliza el JSON devuelto por Codex."""
    if not isinstance(parsed, dict):
        raise ValueError("Estructura JSON inválida en la respuesta de Codex")

    p = parsed
    if (
        not isinstance(p.get("resumen"), str)
        or not isinstance(p.get("requerimientos"), list)
        or not isinstance(p.get("accionables"), list)
        or not isinstance(p.get("decisiones"), list)
    ):
        raise ValueError("Estructura JSON inválida en la respuesta de Codex")

    requerimientos = [
        Requirement(
            id=r["id"],
            descripcion=r["descripcion"],
            prioridad=r["prioridad"],
        )
        for r in p["requerimientos"]
    ]

    # Normalización: fecha null → omitir campo (Accionable.fecha es Optional)
    accionables = [
        Accionable(
            responsable=a["responsable"],
            tarea=a["tarea"],
            fecha=a.get("fecha") or None,
        )
        for a in p["accionables"]
    ]

    return AnalysisResult(
        resumen=p["resumen"],
        requerimientos=requerimientos,
        accionables=accionables,
        decisiones=p["decisiones"],
    )


async def analyze_transcription(
    transcription: list[AttributedSegment],
    _config: PipelineConfig,
) -> AnalysisResult | None:
    """Analiza una transcripción usando Codex y devuelve el resultado estructurado."""
    transcription_text = _build_transcription_text(transcription)
    prompt = _build_prompt(transcription_text)

    return await call_codex_with_schema(
        prompt=prompt,
        schema=_ANALYSIS_SCHEMA,
        validate=_validate_analysis_result,
        max_retries=3,
        timeout_sec=10 * 60,
        error_code="E_CODEX_FAILED",
    )

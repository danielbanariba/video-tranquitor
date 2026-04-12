"""Escritor de notas en formato Obsidian Markdown."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from video_tranquitor.types import AttributedSegment, PipelineConfig, PipelineResult

DEFAULT_VAULT_PATH = "/home/banar/Desktop/obsidian/Farinter/07-Reuniones"


def _format_seconds(total_seconds: float) -> str:
    total = int(total_seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_date_now() -> str:
    """Devuelve la fecha actual como YYYY-MM-DD."""
    d = datetime.now()
    return f"{d.year}-{d.month:02d}-{d.day:02d}"


def _extract_participants(transcription: list[AttributedSegment]) -> list[str]:
    seen: set[str] = set()
    for seg in transcription:
        if seg.speaker:
            seen.add(seg.speaker)
    return sorted(seen)


def _build_frontmatter(
    fecha: str,
    audio_duration_sec: float,
    participants: list[str],
) -> str:
    participants_yaml = (
        "[" + ", ".join(json.dumps(p) for p in participants) + "]"
    )
    lines = [
        "---",
        "tags: [reunion, transcripcion]",
        f"fecha: {fecha}",
        "estado: completado",
        f'duracion: "{_format_seconds(audio_duration_sec)}"',
        f"participantes: {participants_yaml}",
        "---",
    ]
    return "\n".join(lines)


def _build_transcription_section(transcription: list[AttributedSegment]) -> str:
    lines: list[str] = []
    for seg in transcription:
        time = _format_seconds(seg.start)
        if seg.speaker:
            lines.append(f"**{seg.speaker}** ({time}): {seg.text}\n")
        else:
            lines.append(f"({time}): {seg.text}\n")
    return "## Transcripción\n\n" + "\n".join(lines)


def write_obsidian_note(result: PipelineResult, config: PipelineConfig) -> Path:
    """Escribe una nota Markdown en el vault de Obsidian.

    Args:
        result: Resultado completo del pipeline.
        config: Configuración del pipeline (contiene obsidian_vault_path).

    Returns:
        Ruta absoluta del archivo .md generado.

    Raises:
        FileNotFoundError: Si la ruta del vault no existe.
    """
    vault_path = config.obsidian_vault_path or DEFAULT_VAULT_PATH

    if not os.path.exists(vault_path):
        raise FileNotFoundError(
            f"E_VAULT_NOT_FOUND: No se encontró la ruta del vault de Obsidian: {vault_path}"
        )

    fecha = _format_date_now()
    base_name = Path(result.input_file).stem
    file_name = f"{fecha} {base_name}.md"
    output_path = Path(vault_path) / file_name

    participants = _extract_participants(result.transcription)

    sections: list[str] = []

    # Frontmatter
    sections.append(_build_frontmatter(fecha, result.audio_duration_sec, participants))

    # Título
    sections.append(f"\n# Reunión {fecha} - {base_name}\n")

    # Transcripción
    if result.transcription:
        sections.append("\n" + _build_transcription_section(result.transcription))

    # Secciones de análisis (solo si existe)
    analysis = result.analysis
    if analysis:
        if analysis.resumen:
            sections.append("\n## Resumen\n")
            sections.append(analysis.resumen)

        if analysis.requerimientos:
            sections.append("\n## Requerimientos\n")
            req_lines = [
                f"- [ ] **{r.id}** ({r.prioridad}): {r.descripcion}"
                for r in analysis.requerimientos
            ]
            sections.append("\n".join(req_lines))

        if analysis.accionables:
            sections.append("\n## Accionables\n")
            ac_lines: list[str] = []
            for a in analysis.accionables:
                fecha_part = f" — {a.fecha}" if a.fecha else ""
                ac_lines.append(f"- [ ] **{a.responsable}**: {a.tarea}{fecha_part}")
            sections.append("\n".join(ac_lines))

        if analysis.decisiones:
            sections.append("\n## Decisiones\n")
            dec_lines = [f"- {d}" for d in analysis.decisiones]
            sections.append("\n".join(dec_lines))

    content = "\n".join(sections) + "\n"
    output_path.write_text(content, encoding="utf-8")

    return output_path

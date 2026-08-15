"""Análisis de transcripciones mediante Codex para extraer requerimientos y accionables."""

from __future__ import annotations

import asyncio
import json
import logging

from video_tranquitor.llm_client import call_llm_with_schema
from video_tranquitor.types import (
    Accionable,
    AnalysisResult,
    AttributedSegment,
    PipelineConfig,
    Requirement,
)

logger = logging.getLogger(__name__)

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
        "diagrama": {"type": "string"},
    },
    "required": ["resumen", "requerimientos", "accionables", "decisiones", "diagrama"],
    "additionalProperties": False,
}


def _build_transcription_text(transcription: list[AttributedSegment]) -> str:
    """Arma el texto que se le manda al modelo.

    Conserva el prefijo de hablante — es lo que permite atribuir cada accionable
    a un responsable, campo obligatorio del schema. No incluye timestamps: el
    schema no pide marcas de tiempo y el orden ya está implícito en el orden de
    las líneas, así que solo agregarían tokens.
    """
    lines: list[str] = []
    for seg in transcription:
        lines.append(f"{seg.speaker}: {seg.text}" if seg.speaker else seg.text)
    return "\n".join(lines)


def _build_prompt(transcription_text: str) -> str:
    return (
        "Eres un experto en ingeniería de requerimientos. Analiza la siguiente transcripción "
        "de una reunión y extrae información estructurada.\n\n"
        f"Transcripción:\n{transcription_text}\n\n"
        "Devuelve un JSON válido con esta estructura:\n"
        '- resumen: Resumen ejecutivo de la reunión en 1-2 párrafos\n'
        '- requerimientos: array con id ("REQ-001"), descripcion, '
        'prioridad ("alta"|"media"|"baja")\n'
        '- accionables: array con responsable, tarea, fecha opcional (YYYY-MM-DD)\n'
        '- decisiones: array de strings\n'
        "- diagrama: un diagrama Mermaid válido de tipo `flowchart TD` que modele el problema "
        "discutido en la reunión y la solución propuesta o decidida, para entenderlo "
        "visualmente\n\n"
        "Si no hay requerimientos, accionables o decisiones, devuelve arrays vacíos.\n\n"
        "Preservación textual (crítico): copia TAL CUAL, sin parafrasear ni reemplazar por una "
        "descripción genérica, todo identificador técnico que aparezca en la charla: nombres de "
        "host, servidor, base de datos, colección, tabla, campo, archivo, ruta, comando, código "
        "de error, versión y nombre de sistema o herramienta. Si alguien menciona "
        '"mongo-dbcrm-hn", el resumen dice "mongo-dbcrm-hn", no "la base del CRM". Esos datos '
        "son la mitad del valor de la nota: sin ellos nadie puede retomar el trabajo.\n\n"
        "Hablantes: la transcripción viene etiquetada con SPEAKER_00, SPEAKER_01, etc. porque el "
        "diarizador no conoce identidades. Si en la conversación se menciona el nombre real de "
        "quien habla, usalo en responsable y en el resumen (podés aclarar la etiqueta entre "
        "paréntesis). Solo dejá SPEAKER_XX cuando el nombre no aparezca en ningún momento.\n\n"
        "Reglas para el diagrama: devuelve SOLO el cuerpo Mermaid crudo (sin fences ```). "
        "Usa etiquetas de nodo en español y cortas. Entrecomilla toda etiqueta que tenga "
        'espacios, paréntesis o caracteres especiales (p. ej. A["Otra sesión retiene lock"]) '
        "para no romper el parser. Conecta el problema con la solución de forma legible. Si no "
        'hay un problema o solución claro que diagramar, devuelve un string vacío "".'
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

    # diagrama es un campo blando: un diagrama ausente o inválido no debe romper el análisis.
    diagrama = p.get("diagrama", "")
    if not isinstance(diagrama, str):
        diagrama = ""

    return AnalysisResult(
        resumen=p["resumen"],
        requerimientos=requerimientos,
        accionables=accionables,
        decisiones=p["decisiones"],
        diagrama=diagrama,
    )


def _build_consolidation_prompt(analisis: list[AnalysisResult]) -> str:
    """Arma el prompt que consolida varias pasadas sobre la misma reunión."""
    partes = []
    for i, a in enumerate(analisis, 1):
        partes.append(
            f"=== ANÁLISIS {i} ===\n"
            + json.dumps(a.model_dump(), ensure_ascii=False, indent=2)
        )
    cuerpo = "\n\n".join(partes)

    return (
        "Sos un experto en ingeniería de requerimientos. Abajo hay "
        f"{len(analisis)} análisis independientes de LA MISMA reunión, hechos por "
        "el mismo modelo en corridas separadas. Difieren porque el modelo no es "
        "determinista: cada corrida captura detalles que las otras dejan pasar.\n\n"
        "Tu tarea es CONSOLIDARLOS en un único análisis que no pierda nada.\n\n"
        f"{cuerpo}\n\n"
        "Reglas de consolidación:\n"
        "- UNIÓN, no intersección: si un requerimiento, accionable o decisión "
        "aparece en una sola de las versiones, va igual al resultado. Que solo "
        "una corrida lo haya visto no lo hace menos real.\n"
        "- Fusioná los duplicados: si dos versiones describen lo mismo con "
        "distintas palabras, dejá UNA sola entrada y quedate con la redacción "
        "MÁS ESPECÍFICA, la que conserva más datos concretos.\n"
        "- Identificadores técnicos: si una versión escribe el nombre exacto "
        '(por ejemplo "mongo-dbcrm-hn") y otra lo generaliza ("la base del CRM"), '
        "gana SIEMPRE la exacta.\n"
        "- Nombres de personas: si alguna versión identificó el nombre real "
        "detrás de una etiqueta SPEAKER_XX, usá el nombre en todas.\n"
        "- Renumerá los requerimientos de forma correlativa desde REQ-001.\n"
        "- No inventes nada que no esté en ninguna de las versiones.\n"
        "- Para el resumen y el diagrama, elegí la versión más completa y "
        "enriquecela con los datos concretos que aparezcan en las otras.\n\n"
        "Devolvé el resultado con la misma estructura JSON que los análisis de "
        "entrada."
    )


async def _una_pasada(prompt: str, config: PipelineConfig) -> AnalysisResult | None:
    return await call_llm_with_schema(
        prompt=prompt,
        schema=_ANALYSIS_SCHEMA,
        validate=_validate_analysis_result,
        provider=config.analysis_provider,
        model=config.analysis_model,
        effort=config.analysis_effort,
        max_retries=3,
        timeout_sec=10 * 60,
        error_code="E_CODEX_FAILED",
    )


async def analyze_transcription(
    transcription: list[AttributedSegment],
    config: PipelineConfig,
) -> AnalysisResult | None:
    """Analiza una transcripción y devuelve el resultado estructurado.

    Con ``config.analysis_passes`` mayor a 1 corre varias pasadas en paralelo y
    las une con una pasada final de consolidación. El modelo no es determinista:
    se midió que con la misma transcripción de entrada una corrida captura el
    nombre exacto de una base de datos y la siguiente lo omite. Unir varias
    recupera lo que cada una deja pasar.

    Degradación: si algunas pasadas fallan se consolidan las que sobrevivieron;
    si sobrevive una sola se devuelve tal cual; si falla la consolidación se
    devuelve la primera pasada en vez de perder todo el trabajo.
    """
    prompt = _build_prompt(_build_transcription_text(transcription))
    pasadas = max(1, config.analysis_passes)

    if pasadas == 1:
        return await _una_pasada(prompt, config)

    print(f"  Análisis: {pasadas} pasadas en paralelo, después se consolidan...")
    resultados = await asyncio.gather(*(_una_pasada(prompt, config) for _ in range(pasadas)))
    validos = [r for r in resultados if r is not None]

    if not validos:
        return None
    if len(validos) == 1:
        print("  Análisis: solo una pasada devolvió resultado, no hay nada que unir.")
        return validos[0]

    print(f"  Análisis: consolidando {len(validos)} pasadas...")
    consolidado = await _una_pasada(_build_consolidation_prompt(validos), config)
    if consolidado is None:
        logger.warning("La consolidación falló; se usa la primera pasada.")
        return validos[0]
    return consolidado

"""Escritor de salida en formato TOON (Token-Oriented Object Notation).

La librería @toon-format/toon de Node.js no tiene equivalente exacto en PyPI
con la misma API. python-toon implementa la misma especificación TOON pero con
una API diferente. Se implementa el encoding inline para garantizar compatibilidad
exacta con la salida del TS: `encode({ transcripciones: data })`.

El formato tabular TOON para un array de objetos uniformes es:
    clave[N]{campo1,campo2,...}:
      "valor1","valor2",...
      ...
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from video_tranquitor.types import Transcription


def encode(data: dict[str, list[Transcription] | list[dict]]) -> str:
    """Codifica un dict con listas (de Transcription o dicts) al formato TOON tabular.

    Acepta tanto modelos Pydantic Transcription como diccionarios planos con las
    claves inicio/fin/texto (útil para el árbitro de ensemble).

    Produce la misma estructura que @toon-format/toon v2 encode():
        transcripciones[N]{inicio,fin,texto}:
          "00:00:00","00:02:00","texto..."
          ...
    """
    lines: list[str] = []

    for key, items in data.items():
        if not items:
            lines.append(f"{key}[0]{{inicio,fin,texto}}:")
            continue

        lines.append(f"{key}[{len(items)}]{{inicio,fin,texto}}:")
        for item in items:
            if isinstance(item, dict):
                inicio, fin, texto = item["inicio"], item["fin"], item["texto"]
            else:
                inicio, fin, texto = item.inicio, item.fin, item.texto
            buf = io.StringIO()
            writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="")
            writer.writerow([inicio, fin, texto])
            lines.append(f"  {buf.getvalue()}")

    return "\n".join(lines) + "\n"


# Alias interno mantenido por compatibilidad
_encode_toon = encode


def write_toon(data: list[Transcription], output_path: str | Path) -> None:
    """Escribe la lista de transcripciones en formato TOON al archivo indicado.

    Args:
        data:        Lista de objetos Transcription a serializar.
        output_path: Ruta de salida del archivo .toon.
    """
    content = _encode_toon({"transcripciones": data})
    Path(output_path).write_text(content, encoding="utf-8")

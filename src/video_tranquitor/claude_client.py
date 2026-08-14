"""Cliente para ejecutar Claude Code CLI con validación de esquema JSON.

Estructura paralela a codex_client.py: shell-out al CLI, retry con backoff
exponencial, extracción y validación con _extract_json. Claude soporta
--json-schema nativo (equivalente al --output-schema de Codex), por lo que
la respuesta llega ya estructurada.

Se usa como fallback automático cuando Codex agota retries (ver codex_client).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
from typing import Callable, TypeVar

from video_tranquitor.codex_client import _extract_json

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SEC = 20 * 60
DEFAULT_ERROR_CODE = "E_CLAUDE_FAILED"
INITIAL_BACKOFF_SEC = 2.0


def _call_claude(
    prompt: str,
    schema: object,
    timeout_sec: float,
    model: str = "",
    effort: str = "",
) -> str:
    """Ejecuta `claude -p` con el prompt en stdin y schema vía --json-schema.

    Si `model` está vacío, Claude Code usa el modelo por defecto de la suscripción.
    `effort` acepta low | medium | high | xhigh | max (no soporta `minimal`).
    """
    schema_json = json.dumps(schema)
    model_args = ["--model", model] if model else []
    effort_args = ["--effort", effort] if effort else []

    proc = subprocess.Popen(
        [
            "claude",
            "-p",
            *model_args,
            *effort_args,
            "--json-schema", schema_json,
            "--tools", "",
            "--no-session-persistence",
            "--output-format", "text",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = proc.communicate(
            input=prompt.encode("utf-8"),
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise TimeoutError(f"Claude timeout ({timeout_sec}s)")

    if proc.returncode != 0:
        stderr_tail = (stderr or b"").decode("utf-8", errors="replace")[-500:]
        raise RuntimeError(f"Claude exit {proc.returncode}: {stderr_tail}")

    content = stdout.decode("utf-8", errors="replace").strip()
    if not content:
        raise RuntimeError("Respuesta vacía de Claude")

    return content


async def call_claude_with_schema(
    prompt: str,
    schema: object,
    validate: Callable[[object], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    error_code: str = DEFAULT_ERROR_CODE,
    model: str = "",
    effort: str = "",
) -> T | None:
    """Ejecuta Claude Code CLI con un esquema JSON, valida la respuesta y reintenta.

    Mismo contrato que call_codex_with_schema: devuelve el objeto validado
    de tipo T, o None si todos los intentos fallaron.

    Args:
        model:  Modelo de Claude Code. Vacío = el default de la suscripción.
        effort: Nivel de razonamiento. Vacío = el default del CLI.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = await asyncio.to_thread(
                _call_claude, prompt, schema, timeout_sec, model, effort
            )
            json_text = _extract_json(raw_output)
            parsed = json.loads(json_text)
            return validate(parsed)
        except Exception as error:
            last_error = error

            if attempt < max_retries:
                backoff = INITIAL_BACKOFF_SEC * (2 ** (attempt - 1))
                print(
                    f"  Claude: intento {attempt}/{max_retries} fallido. "
                    f"Reintentando en {backoff:.0f}s..."
                )
                await asyncio.sleep(backoff)

    logger.error(
        "%s: Claude fallido tras %d intentos. %s",
        error_code,
        max_retries,
        str(last_error),
    )
    return None

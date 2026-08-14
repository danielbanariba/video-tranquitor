"""Cliente para ejecutar Codex con validación de esquema JSON."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT_SEC = 20 * 60  # 20 minutos
DEFAULT_ERROR_CODE = "E_CODEX_FAILED"
INITIAL_BACKOFF_SEC = 2.0

# Niveles de esfuerzo que aceptan tanto Codex como Claude Code.
SHARED_EFFORT_LEVELS = frozenset({"low", "medium", "high", "xhigh"})


def _extract_json(raw: str) -> str:
    """Extrae el bloque JSON de la respuesta cruda de Codex."""
    trimmed = raw.strip()

    # Respuesta directa JSON
    if trimmed.startswith("{") and trimmed.endswith("}"):
        return trimmed

    # Bloque de código con triple backtick
    import re
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", trimmed)
    if fence_match:
        return fence_match.group(1).strip()

    # Extracción por posición de llaves
    first_brace = trimmed.find("{")
    last_brace = trimmed.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        return trimmed[first_brace : last_brace + 1]

    return trimmed


def _call_codex(
    prompt: str,
    schema: object,
    timeout_sec: float,
    model: str = "",
    effort: str = "",
) -> str:
    """Ejecuta `codex exec` con el prompt en stdin y devuelve el contenido del archivo de salida.

    Si `model` está vacío, Codex usa el modelo por defecto de ~/.codex/config.toml.
    `effort` acepta minimal | low | medium | high | xhigh (no soporta `max`).
    """
    tmp_dir = tempfile.mkdtemp(prefix="codex-client-")
    schema_path = os.path.join(tmp_dir, "schema.json")
    output_path = os.path.join(tmp_dir, "output.txt")

    try:
        with open(schema_path, "w", encoding="utf-8") as f:
            json.dump(schema, f)

        model_args = ["--model", model] if model else []
        effort_args = ["-c", f"model_reasoning_effort={effort}"] if effort else []

        proc = subprocess.Popen(
            [
                "codex",
                "exec",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                *model_args,
                *effort_args,
                "--output-schema", schema_path,
                "--output-last-message", output_path,
                "--color", "never",
                "-",
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
            raise TimeoutError(f"Codex timeout ({timeout_sec}s)")

        if proc.returncode != 0:
            stderr_tail = (stderr or b"").decode("utf-8", errors="replace")[-500:]
            raise RuntimeError(f"Codex exit {proc.returncode}: {stderr_tail}")

        if not os.path.exists(output_path):
            raise RuntimeError("Codex no generó archivo de salida")

        with open(output_path, encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            raise RuntimeError("Respuesta vacía de Codex")

        return content

    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


async def call_codex_with_schema[T](
    prompt: str,
    schema: object,
    validate: Callable[[object], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    error_code: str = DEFAULT_ERROR_CODE,
    model: str = "",
    effort: str = "",
) -> T | None:
    """Ejecuta Codex con un esquema JSON, valida la respuesta y reintenta en caso de fallo.

    Args:
        prompt:      Texto del prompt enviado a Codex via stdin.
        schema:      JSON Schema que Codex usa para estructurar la respuesta.
        validate:    Función que parsea y valida el objeto Python resultante.
        max_retries: Cantidad máxima de intentos (default 3).
        timeout_sec: Tiempo límite por intento en segundos (default 1200).
        error_code:  Prefijo del mensaje de error final.
        model:       Modelo de Codex a usar. Vacío = el default de ~/.codex/config.toml.
        effort:      Nivel de razonamiento. Vacío = el default de config.toml.

    Returns:
        El objeto validado de tipo T, o None si todos los intentos fallaron.
    """
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_output = await asyncio.to_thread(
                _call_codex, prompt, schema, timeout_sec, model, effort
            )
            json_text = _extract_json(raw_output)
            parsed = json.loads(json_text)
            return validate(parsed)
        except Exception as error:
            last_error = error

            if attempt < max_retries:
                backoff = INITIAL_BACKOFF_SEC * (2 ** (attempt - 1))
                print(
                    f"  Codex: intento {attempt}/{max_retries} fallido. "
                    f"Reintentando en {backoff:.0f}s..."
                )
                await asyncio.sleep(backoff)

    logger.error(
        "%s: Codex fallido tras %d intentos. %s",
        error_code,
        max_retries,
        str(last_error),
    )

    # Fallback automático a Claude Code CLI si está disponible en PATH.
    # Cubre quota exhaustion, timeouts y cualquier otro fallo persistente.
    if shutil.which("claude"):
        print("  Codex agotó retries. Probando con Claude Code CLI como fallback...")
        from video_tranquitor.claude_client import call_claude_with_schema

        # `model` NO se reenvía: un ID de OpenAI es inválido en Claude Code.
        # `effort` solo si el nivel existe en ambos CLIs (Codex tiene `minimal`,
        # Claude tiene `max`; ninguno acepta el exclusivo del otro).
        shared_effort = effort if effort in SHARED_EFFORT_LEVELS else ""

        return await call_claude_with_schema(
            prompt=prompt,
            schema=schema,
            validate=validate,
            max_retries=max_retries,
            timeout_sec=timeout_sec,
            error_code=error_code,
            effort=shared_effort,
        )

    return None

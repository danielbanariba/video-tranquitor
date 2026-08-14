"""Selector de proveedor para las llamadas con esquema JSON.

Despacha a `codex_client` o `claude_client` según `config.analysis_provider`,
manteniendo el mismo contrato: devuelve el objeto validado o None.

- `codex`  (default): usa `codex exec`. Si agota retries cae automáticamente
  a Claude Code CLI (ver codex_client), así que sigue siendo la opción más
  resistente cuando ambos CLIs están disponibles.
- `claude`: va directo a Claude Code CLI, sin pasar por Codex. Pensado para
  cuando no hay cuota de ChatGPT y esperar los retries de Codex es tiempo
  perdido.
"""

from __future__ import annotations

from collections.abc import Callable

from video_tranquitor.codex_client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SEC,
    call_codex_with_schema,
)

PROVIDERS = ("codex", "claude")


async def call_llm_with_schema[T](
    prompt: str,
    schema: object,
    validate: Callable[[object], T],
    provider: str = "codex",
    model: str = "",
    effort: str = "",
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    error_code: str = "E_LLM_FAILED",
) -> T | None:
    """Ejecuta el proveedor configurado con un esquema JSON y valida la respuesta.

    Args:
        prompt:      Texto del prompt enviado al CLI via stdin.
        schema:      JSON Schema que estructura la respuesta.
        validate:    Función que parsea y valida el objeto Python resultante.
        provider:    "codex" o "claude".
        model:       Modelo del proveedor elegido. Vacío = su default.
        effort:      Nivel de razonamiento. Vacío = el default del proveedor.
        max_retries: Cantidad máxima de intentos.
        timeout_sec: Tiempo límite por intento en segundos.
        error_code:  Prefijo del mensaje de error final.

    Returns:
        El objeto validado de tipo T, o None si todos los intentos fallaron.
    """
    if provider == "claude":
        # Importación tardía: evita cargar el cliente si no se usa.
        from video_tranquitor.claude_client import call_claude_with_schema  # noqa: PLC0415

        return await call_claude_with_schema(
            prompt=prompt,
            schema=schema,
            validate=validate,
            max_retries=max_retries,
            timeout_sec=timeout_sec,
            error_code=error_code,
            model=model,
            effort=effort,
        )

    return await call_codex_with_schema(
        prompt=prompt,
        schema=schema,
        validate=validate,
        max_retries=max_retries,
        timeout_sec=timeout_sec,
        error_code=error_code,
        model=model,
        effort=effort,
    )

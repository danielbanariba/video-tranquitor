"""Interfaz de línea de comandos del pipeline de transcripción."""

from __future__ import annotations

import asyncio
import sys

import click

from video_tranquitor.config import load_config
from video_tranquitor.pipeline import run_pipeline
from video_tranquitor.watcher import start_watcher


@click.command()
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True),
    default=None,
    help="Procesar un archivo específico",
)
@click.option(
    "--watch",
    "watch_mode",
    is_flag=True,
    default=False,
    help="Modo watcher (daemon)",
)
@click.argument(
    "positional",
    required=False,
    type=click.Path(exists=True),
    default=None,
)
def main(
    file_path: str | None,
    watch_mode: bool,
    positional: str | None,
) -> None:
    """Pipeline de transcripción y análisis de audio/video."""

    # Prioridad: --file > argumento posicional > --watch (modo default)
    target_file: str | None = file_path or positional

    try:
        config = load_config()
    except ValueError as exc:
        click.echo(f"Error de configuración: {exc}", err=True)
        sys.exit(1)

    if target_file:
        # Modo de archivo único
        try:
            result = asyncio.run(run_pipeline(target_file, config))
            click.echo(
                f"\nProcesamiento completado. Etapas ejecutadas: {', '.join(result.stages_run)}"
            )
        except Exception as exc:  # noqa: BLE001
            click.echo(f"Error durante el procesamiento: {exc}", err=True)
            sys.exit(1)
    else:
        # Modo watcher (también cuando se pasa --watch o no se pasa nada)
        start_watcher(config, lambda path: run_pipeline(path, config))

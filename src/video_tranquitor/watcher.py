"""Watcher de directorio para detectar archivos nuevos de audio/video."""

from __future__ import annotations

import asyncio
import os
import queue
import signal
import threading
from pathlib import Path
from typing import Callable, Coroutine, Any

from watchdog.events import FileCreatedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from video_tranquitor.pipeline import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from video_tranquitor.types import PipelineConfig

DEBOUNCE_S = 0.5


class _Handler(FileSystemEventHandler):
    """Manejador de eventos del sistema de archivos."""

    def __init__(self, file_queue: "queue.Queue[str]") -> None:
        super().__init__()
        self._queue = file_queue
        self._timers: dict[str, threading.Timer] = {}

    def on_created(self, event: FileCreatedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return

        path = event.src_path
        name = Path(path).name
        ext = Path(path).suffix.lower()

        # Ignorar archivos ocultos y temporales
        if name.startswith(".") or name.startswith("temp_"):
            return

        # Solo procesar extensiones soportadas
        if ext not in VIDEO_EXTENSIONS and ext not in AUDIO_EXTENSIONS:
            return

        # Debounce: cancelar timer previo si existe
        existing = self._timers.get(path)
        if existing:
            existing.cancel()

        def _fire() -> None:
            self._timers.pop(path, None)
            self._queue.put(path)

        timer = threading.Timer(DEBOUNCE_S, _fire)
        self._timers[path] = timer
        timer.start()


def start_watcher(
    config: PipelineConfig,
    on_file: Callable[[str], Coroutine[Any, Any, None]],
) -> None:
    """Inicia el watcher en modo daemon.

    Monitorea ``config.watch_dir`` y llama a ``on_file(path)`` cada vez que
    se detecta un archivo de audio/video nuevo.

    Los eventos de watchdog llegan desde un hilo secundario. Para manejar
    correctamente las corrutinas, se usa una ``queue.Queue`` como puente:
    el hilo del watcher encola rutas y el loop principal las drena.

    Args:
        config:  Configuración del pipeline.
        on_file: Corrutina asíncrona que procesa cada archivo detectado.
    """
    watch_dir = config.watch_dir

    if not os.path.exists(watch_dir):
        os.makedirs(watch_dir, exist_ok=True)
        print(f"Directorio creado: {watch_dir}")

    file_queue: queue.Queue[str] = queue.Queue()
    handler = _Handler(file_queue)
    observer = Observer()
    observer.schedule(handler, watch_dir, recursive=False)
    observer.start()

    print(f"Watcher iniciado. Monitoreando: {watch_dir}")
    print("Esperando archivos de audio/video... (Ctrl+C para detener)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(signum: int, frame: object) -> None:  # noqa: ARG001
        print("\nDeteniendo watcher...")
        observer.stop()
        loop.call_soon_threadsafe(loop.stop)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    async def _drain() -> None:
        while True:
            try:
                path = file_queue.get_nowait()
                print(f"\nArchivo detectado: {path}")
                try:
                    await on_file(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"Error al procesar {path}: {exc}")
            except queue.Empty:
                await asyncio.sleep(0.1)

    try:
        loop.run_until_complete(_drain())
    finally:
        observer.join()
        loop.close()
        print("Watcher detenido.")

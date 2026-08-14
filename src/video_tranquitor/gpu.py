"""Liberación de la VRAM que PyTorch mantiene cacheada entre etapas."""

from __future__ import annotations

import gc
import logging
import sys

logger = logging.getLogger(__name__)


def release_gpu_memory() -> bool:
    """Devuelve al driver la VRAM que PyTorch tiene reservada pero sin usar.

    El caching allocator de PyTorch no le devuelve la memoria al driver cuando
    se liberan los modelos: la retiene reservada mientras viva el proceso. Para
    un one-shot da igual, pero en modo watcher el daemon queda ocupando la GPU
    entre archivo y archivo e impide que corra cualquier otra cosa.

    No importa torch si todavía no está cargado: hacerlo solo para limpiar
    costaría segundos de arranque y no habría nada reservado que liberar.

    Returns:
        True si se vació la caché, False si no había nada que hacer o si falló.
    """
    torch = sys.modules.get("torch")
    if torch is None:
        return False

    gc.collect()

    try:
        if not torch.cuda.is_available():
            return False
        torch.cuda.empty_cache()
    except Exception as error:  # noqa: BLE001 — housekeeping, nunca debe romper el pipeline
        logger.debug("No se pudo liberar la VRAM: %s", error)
        return False

    return True

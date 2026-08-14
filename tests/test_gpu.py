"""Tests para video_tranquitor.gpu.release_gpu_memory."""

from __future__ import annotations

import sys
import types

import pytest

from video_tranquitor.gpu import release_gpu_memory


@pytest.fixture
def sin_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "torch", raising=False)


def fake_torch(*, cuda_available: bool, on_empty_cache=None) -> types.ModuleType:
    """Construye un módulo torch falso con lo mínimo que usa release_gpu_memory."""
    modulo = types.ModuleType("torch")
    llamadas: list[str] = []

    def empty_cache() -> None:
        llamadas.append("empty_cache")
        if on_empty_cache is not None:
            on_empty_cache()

    modulo.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: cuda_available,
        empty_cache=empty_cache,
    )
    modulo.llamadas = llamadas  # type: ignore[attr-defined]
    return modulo


class TestReleaseGpuMemory:
    # Clave del diseño: no importar torch solo para limpiar. Si no está cargado
    # no hay nada reservado, e importarlo costaría segundos de arranque.
    def test_does_not_import_torch_when_not_loaded(self, sin_torch: None) -> None:
        assert release_gpu_memory() is False
        assert "torch" not in sys.modules

    def test_returns_false_when_cuda_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_available=False))

        assert release_gpu_memory() is False

    def test_empties_cache_when_cuda_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        modulo = fake_torch(cuda_available=True)
        monkeypatch.setitem(sys.modules, "torch", modulo)

        assert release_gpu_memory() is True
        assert modulo.llamadas == ["empty_cache"]  # type: ignore[attr-defined]

    # Liberar memoria nunca debe tumbar el pipeline: es housekeeping.
    def test_swallows_errors_from_empty_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def explota() -> None:
            raise RuntimeError("CUDA driver error")

        monkeypatch.setitem(
            sys.modules, "torch", fake_torch(cuda_available=True, on_empty_cache=explota)
        )

        assert release_gpu_memory() is False

"""Tests para video_tranquitor.watcher — apagado del daemon."""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable

import pytest

from video_tranquitor import watcher as watcher_mod
from video_tranquitor.types import PipelineConfig


@pytest.fixture
def config(tmp_path) -> PipelineConfig:
    return PipelineConfig(
        watch_dir=str(tmp_path),
        output_dir=str(tmp_path / "output"),
        transcriber="local",
        whisperx_model="large-v3",
        whisper_cpp_path="/no/existe",
        whisper_model_path="/no/existe",
        enable_diarization=False,
        enable_analysis=False,
        enable_obsidian=False,
        enable_toon=False,
        obsidian_vault_path="",
        hf_token="",
        openai_api_key="",
        audio_filter="",
        language="es",
        transcription_prompt="",
        transcribe_model="gpt-4o-transcribe",
        target_sample_rate=16000,
    )


class TestApagado:
    # Ctrl+C frenaba el loop con loop.stop() mientras _drain seguía vivo, y
    # run_until_complete reventaba con "Event loop stopped before Future
    # completed": traceback y exit code 1 en cada apagado normal del daemon.
    def test_ctrl_c_detiene_el_watcher_sin_excepcion(
        self, config: PipelineConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handlers: dict[int, Callable[[int, object], None]] = {}
        registrado = threading.Event()

        # Capturar el handler en vez de instalarlo: una señal real que llegue
        # antes de que start_watcher lo registre mataría la corrida de pytest.
        def fake_signal(signum: int, handler: Callable[[int, object], None]) -> None:
            handlers[signum] = handler
            if signum == signal.SIGINT:
                registrado.set()

        monkeypatch.setattr(watcher_mod.signal, "signal", fake_signal)

        def ctrl_c() -> None:
            registrado.wait(timeout=5)
            handlers[signal.SIGINT](signal.SIGINT, None)

        threading.Thread(target=ctrl_c, daemon=True).start()

        async def on_file(_path: str) -> None:
            raise AssertionError("no debería procesarse ningún archivo")

        watcher_mod.start_watcher(config, on_file)

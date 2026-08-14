"""Tests para video_tranquitor.pipeline — limpieza del WAV temporal."""

from __future__ import annotations

import os

import pytest

from video_tranquitor import pipeline as pipeline_mod
from video_tranquitor.pipeline import run_pipeline
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


def _wav_temporal(config: PipelineConfig, nombre: str) -> str:
    return os.path.join(config.output_dir, f"temp_{nombre}.wav")


class TestLimpiezaDelWavTemporal:
    # El caso que se vio en producción: el ensemble murió por falta de VRAM y
    # dejó un WAV de casi 100 MB en output/. El watcher lo ignora por el prefijo
    # temp_, así que nadie lo limpiaba nunca.
    async def test_borra_el_wav_cuando_la_transcripcion_falla(
        self, config: PipelineConfig, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entrada = tmp_path / "reunion.wav"
        entrada.write_bytes(b"RIFF")
        wav = _wav_temporal(config, "reunion")

        def fake_preprocess(_src, destino, _filtro, _sr):
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino, "wb") as f:
                f.write(b"x" * 1024)
            return True

        monkeypatch.setattr(pipeline_mod, "preprocess_audio", fake_preprocess)
        monkeypatch.setattr(pipeline_mod, "get_audio_duration", lambda _p: 10.0)

        def explota(*_a, **_k):
            raise RuntimeError("E_WHISPER_NOT_FOUND: simulando fallo de transcripción")

        monkeypatch.setattr(pipeline_mod, "transcribe_local", explota)

        with pytest.raises(RuntimeError, match="E_WHISPER_NOT_FOUND"):
            await run_pipeline(str(entrada), config)

        assert not os.path.exists(wav), "el WAV temporal quedó colgado tras el fallo"

    # Si el preprocess falla dejando un archivo a medias, tampoco debe quedar.
    async def test_borra_el_wav_cuando_el_preprocess_falla(
        self, config: PipelineConfig, tmp_path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entrada = tmp_path / "reunion.wav"
        entrada.write_bytes(b"RIFF")
        wav = _wav_temporal(config, "reunion")

        def preprocess_a_medias(_src, destino, _filtro, _sr):
            os.makedirs(os.path.dirname(destino), exist_ok=True)
            with open(destino, "wb") as f:
                f.write(b"parcial")
            return False

        monkeypatch.setattr(pipeline_mod, "preprocess_audio", preprocess_a_medias)

        with pytest.raises(RuntimeError, match="No se pudo preprocesar"):
            await run_pipeline(str(entrada), config)

        assert not os.path.exists(wav), "el WAV parcial quedó colgado"

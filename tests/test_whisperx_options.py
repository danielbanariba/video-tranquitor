"""Tests para las opciones que video_tranquitor le pasa a WhisperX."""

from __future__ import annotations

import sys
import types

import pytest

from video_tranquitor.types import PipelineConfig


def _config(**overrides) -> PipelineConfig:
    base = {
        "watch_dir": ".",
        "output_dir": "./output",
        "transcriber": "whisperx",
        "whisperx_model": "large-v3",
        "whisper_cpp_path": "",
        "whisper_model_path": "",
        "enable_diarization": False,
        "enable_analysis": False,
        "enable_obsidian": False,
        "enable_toon": False,
        "obsidian_vault_path": "",
        "hf_token": "",
        "openai_api_key": "",
        "audio_filter": "",
        "language": "es",
        "transcription_prompt": "Mantiene nombres propios, numeros y siglas.",
        "transcribe_model": "gpt-4o-transcribe",
        "target_sample_rate": 16000,
    }
    base.update(overrides)
    return PipelineConfig(**base)


@pytest.fixture
def whisperx_falso(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Reemplaza el módulo whisperx por un doble que registra cómo lo llaman."""
    registro: dict = {}

    class ModeloFalso:
        def transcribe(self, _audio, **kwargs):
            registro["transcribe_kwargs"] = kwargs
            return {"segments": [{"text": "hola", "start": 0.0, "end": 1.0, "words": []}]}

    mod = types.ModuleType("whisperx")
    mod.load_model = lambda *a, **kw: (registro.update(load_model_kwargs=kw), ModeloFalso())[1]
    mod.load_audio = lambda _p: object()
    mod.load_align_model = lambda **_kw: (_ for _ in ()).throw(RuntimeError("sin alineación"))
    mod.align = lambda *a, **kw: {}

    torch_falso = types.ModuleType("torch")
    torch_falso.cuda = types.SimpleNamespace(
        is_available=lambda: False, empty_cache=lambda: None
    )

    monkeypatch.setitem(sys.modules, "whisperx", mod)
    monkeypatch.setitem(sys.modules, "torch", torch_falso)
    return registro


class TestOpcionesDeWhisperX:
    # NO se pasa initial_prompt, y es una decisión medida, no un olvido. Sobre
    # 600 s del audio real, pasarlo bajó las palabras de 880 a 735 y perdió
    # menciones de Honduras, Mongo y Power Query. Este test existe para que el
    # día que alguien lo "arregle", falle y vaya a leer por qué.
    def test_no_pasa_initial_prompt_aunque_este_configurado(
        self, whisperx_falso: dict
    ) -> None:
        from video_tranquitor.transcribers.whisperx import transcribe_whisperx

        transcribe_whisperx("audio.wav", _config(), "large-v3")

        opciones = whisperx_falso["load_model_kwargs"]["asr_options"]
        assert "initial_prompt" not in opciones

    def test_beam_size_por_defecto_es_el_de_whisperx(self, whisperx_falso: dict) -> None:
        from video_tranquitor.transcribers.whisperx import transcribe_whisperx

        transcribe_whisperx("audio.wav", _config(), "large-v3")

        opciones = whisperx_falso["load_model_kwargs"]["asr_options"]
        assert opciones["beam_size"] == 5
        assert opciones["best_of"] == 5

    # Configurable para poder volver a medirlo, pero subirlo NO es gratis:
    # con beam 10 el modelo perdió DLFarInter por completo (4 menciones -> 0).
    def test_beam_size_es_configurable(self, whisperx_falso: dict) -> None:
        from video_tranquitor.transcribers.whisperx import transcribe_whisperx

        transcribe_whisperx("audio.wav", _config(whisperx_beam_size=10), "large-v3")

        opciones = whisperx_falso["load_model_kwargs"]["asr_options"]
        assert opciones["beam_size"] == 10
        assert opciones["best_of"] == 10

"""Tests para video_tranquitor.preprocessor — diagnóstico de fallos."""

from __future__ import annotations

import subprocess

import pytest

from video_tranquitor import preprocessor as pre


class TestPreprocessAudio:
    # Una etapa que tarda dos minutos y falla tiene que decir POR QUÉ.
    # Antes se capturaba el stderr de ffmpeg y se descartaba.
    def test_incluye_el_stderr_de_ffmpeg_en_el_log(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def ffmpeg_falla(_cmd, **_kw):
            return subprocess.CompletedProcess(
                _cmd, 1, stdout=b"", stderr=b"Unknown encoder 'afftdn'"
            )

        monkeypatch.setattr(pre.subprocess, "run", ffmpeg_falla)

        with caplog.at_level("ERROR"):
            ok = pre.preprocess_audio("in.mp4", "out.wav", "", 16000)

        assert ok is False
        assert "Unknown encoder 'afftdn'" in caplog.text

    def test_reintenta_sin_filtros_y_lo_dice(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        intentos: list[list[str]] = []

        def ffmpeg(cmd, **_kw):
            intentos.append(cmd)
            # Falla con filtros, funciona sin ellos
            falla = "-af" in cmd
            return subprocess.CompletedProcess(
                cmd, 1 if falla else 0, stdout=b"", stderr=b"filtro invalido"
            )

        monkeypatch.setattr(pre.subprocess, "run", ffmpeg)

        with caplog.at_level("ERROR"):
            ok = pre.preprocess_audio("in.mp4", "out.wav", "afftdn=nf=-25", 16000)

        assert ok is True
        assert len(intentos) == 2
        assert "filtro invalido" in caplog.text


class TestGetAudioDuration:
    # Devolver 0.0 en silencio hace que el pipeline reporte "00:00:00" y siga
    # como si nada, escondiendo un ffprobe roto.
    def test_avisa_cuando_ffprobe_falla(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        def ffprobe_falla(*_a, **_kw):
            raise FileNotFoundError("ffprobe no está en PATH")

        monkeypatch.setattr(pre.subprocess, "run", ffprobe_falla)

        with caplog.at_level("WARNING"):
            duracion = pre.get_audio_duration("audio.wav")

        assert duracion == 0.0
        assert "ffprobe" in caplog.text

    def test_devuelve_la_duracion_cuando_funciona(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            pre.subprocess,
            "run",
            lambda *_a, **_kw: subprocess.CompletedProcess([], 0, stdout="123.45\n", stderr=""),
        )

        assert pre.get_audio_duration("audio.wav") == pytest.approx(123.45)

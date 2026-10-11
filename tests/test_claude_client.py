"""Tests para video_tranquitor.claude_client — reporte de fallos del CLI."""

from __future__ import annotations

import logging

import pytest

from video_tranquitor import claude_client as claude_mod
from video_tranquitor.claude_client import call_claude_with_schema


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes) -> None:
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    def communicate(self, input: bytes | None = None, timeout: float | None = None):
        return self._stdout, self._stderr


def _popen_returning(proc: _FakeProc):
    def fake_popen(*args, **kwargs) -> _FakeProc:
        return proc

    return fake_popen


class TestFalloDelCli:
    async def test_reporta_el_error_que_claude_escribe_en_stdout(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Claude Code escribe el límite de uso en stdout y deja stderr vacío.
        proc = _FakeProc(1, b"You've hit your weekly limit\n", b"")
        monkeypatch.setattr(claude_mod.subprocess, "Popen", _popen_returning(proc))

        with caplog.at_level(logging.ERROR, logger=claude_mod.__name__):
            result = await call_claude_with_schema(
                prompt="p", schema={}, validate=lambda x: x, max_retries=1
            )

        assert result is None
        assert "You've hit your weekly limit" in caplog.text

    async def test_conserva_stderr_cuando_el_cli_lo_usa(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        proc = _FakeProc(2, b"", b"error: unknown option '--foo'\n")
        monkeypatch.setattr(claude_mod.subprocess, "Popen", _popen_returning(proc))

        with caplog.at_level(logging.ERROR, logger=claude_mod.__name__):
            await call_claude_with_schema(
                prompt="p", schema={}, validate=lambda x: x, max_retries=1
            )

        assert "unknown option '--foo'" in caplog.text

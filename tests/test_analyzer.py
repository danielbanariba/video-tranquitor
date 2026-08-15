"""Tests para video_tranquitor.analyzer — validación de resultados y armado del prompt."""

from __future__ import annotations

import pytest

from video_tranquitor import analyzer as analyzer_mod
from video_tranquitor.analyzer import (
    _build_prompt,
    _build_transcription_text,
    _validate_analysis_result,
    analyze_transcription,
)
from video_tranquitor.types import (
    AnalysisResult,
    AttributedSegment,
    PipelineConfig,
    Requirement,
)


def _segmento() -> AttributedSegment:
    return AttributedSegment(speaker="SPEAKER_00", text="hola", start=0.0, end=1.0)


def _config(passes: int) -> PipelineConfig:
    return PipelineConfig(
        watch_dir=".",
        output_dir="./output",
        transcriber="whisperx",
        whisperx_model="large-v3",
        whisper_cpp_path="",
        whisper_model_path="",
        enable_diarization=False,
        enable_analysis=True,
        enable_obsidian=False,
        enable_toon=False,
        obsidian_vault_path="",
        hf_token="",
        openai_api_key="",
        audio_filter="",
        language="es",
        transcription_prompt="",
        transcribe_model="gpt-4o-transcribe",
        analysis_passes=passes,
        target_sample_rate=16000,
    )


def base_parsed(**overrides: object) -> dict[str, object]:
    """Construye un dict válido tal como lo devolvería Codex, con overrides opcionales."""
    parsed: dict[str, object] = {
        "resumen": "Resumen ejecutivo de la reunion.",
        "requerimientos": [
            {"id": "REQ-001", "descripcion": "Login con OAuth", "prioridad": "alta"}
        ],
        "accionables": [{"responsable": "Ana", "tarea": "Configurar CI/CD", "fecha": "2026-04-15"}],
        "decisiones": ["Migrar a PostgreSQL"],
    }
    parsed.update(overrides)
    return parsed


class TestValidateAnalysisResultDiagrama:
    def test_passes_diagrama_through_onto_result(self) -> None:
        diagrama = 'flowchart TD\n    A["Problema"] --> B["Solución"]'
        result = _validate_analysis_result(base_parsed(diagrama=diagrama))

        assert result.diagrama == diagrama

    def test_defaults_diagrama_to_empty_string_when_absent(self) -> None:
        result = _validate_analysis_result(base_parsed())

        assert result.diagrama == ""

    def test_coerces_diagrama_to_empty_string_when_wrong_type(self) -> None:
        result = _validate_analysis_result(base_parsed(diagrama=123))

        assert result.diagrama == ""


def make_segment(
    text: str, start: float, end: float, speaker: str | None = None
) -> AttributedSegment:
    return AttributedSegment(speaker=speaker, text=text, start=start, end=end)


class TestBuildTranscriptionText:
    # El hablante se conserva: es lo que permite llenar `responsable` en los accionables.
    def test_keeps_speaker_prefix(self) -> None:
        segments = [
            make_segment("Necesitamos doble factor.", 0.0, 6.0, "SPEAKER_00"),
            make_segment("Yo me encargo del SMS.", 6.0, 11.0, "SPEAKER_01"),
        ]

        result = _build_transcription_text(segments)

        assert result == (
            "SPEAKER_00: Necesitamos doble factor.\nSPEAKER_01: Yo me encargo del SMS."
        )

    # El timestamp NO se envía: el schema no pide marcas de tiempo y el orden
    # ya está implícito en el orden de las líneas.
    def test_omits_timestamps(self) -> None:
        segments = [make_segment("Hola.", 3725.0, 3730.0, "SPEAKER_02")]

        result = _build_transcription_text(segments)

        assert "01:02:05" not in result
        assert "(" not in result
        assert result == "SPEAKER_02: Hola."

    # Sin diarización no hay prefijo: solo el texto limpio.
    def test_without_speaker_emits_bare_text(self) -> None:
        segments = [
            make_segment("Primera línea.", 0.0, 2.0),
            make_segment("Segunda línea.", 2.0, 4.0),
        ]

        result = _build_transcription_text(segments)

        assert result == "Primera línea.\nSegunda línea."

    def test_empty_transcription_returns_empty_string(self) -> None:
        assert _build_transcription_text([]) == ""


class TestBuildPrompt:
    # Los identificadores técnicos son justamente lo que el modelo pierde entre
    # corridas (se midió: `mongo-dbcrm-hn` apareció en una y no en la siguiente,
    # con transcripción de entrada idéntica).
    def test_instructs_verbatim_technical_identifiers(self) -> None:
        prompt = _build_prompt("SPEAKER_00: la base es mongo-dbcrm-hn en el servidor 202.")

        assert "textual" in prompt.lower()
        for termino in ("host", "base de datos", "colección", "ruta", "comando"):
            assert termino in prompt.lower(), f"falta '{termino}' en las reglas de preservación"

    # Si en la charla se nombra a la persona, el accionable debe llevar el nombre
    # real y no la etiqueta anónima del diarizador.
    def test_instructs_resolving_speaker_labels_to_names(self) -> None:
        prompt = _build_prompt("SPEAKER_02: yo lo hago, dice Luis.")

        assert "SPEAKER_" in prompt
        assert "nombre real" in prompt.lower()

    def test_still_embeds_the_transcription(self) -> None:
        prompt = _build_prompt("SPEAKER_00: hola.")

        assert "SPEAKER_00: hola." in prompt


# ---------------------------------------------------------------------
# Análisis multi-pasada
# ---------------------------------------------------------------------


def _resultado(resumen: str, n_req: int = 1) -> AnalysisResult:
    return AnalysisResult(
        resumen=resumen,
        requerimientos=[
            Requirement(id=f"REQ-{i:03d}", descripcion=f"req {i}", prioridad="alta")
            for i in range(1, n_req + 1)
        ],
        accionables=[],
        decisiones=[],
        diagrama="",
    )


class TestAnalisisMultiPasada:
    async def test_una_sola_pasada_no_consolida(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llamadas: list[str] = []

        async def fake_llm(*, prompt: str, **_kw):
            llamadas.append(prompt)
            return _resultado("único")

        monkeypatch.setattr(analyzer_mod, "call_llm_with_schema", fake_llm)
        config = _config(passes=1)

        resultado = await analyze_transcription([_segmento()], config)

        assert resultado is not None and resultado.resumen == "único"
        assert len(llamadas) == 1, "con una pasada no debe haber consolidación"

    async def test_varias_pasadas_mas_consolidacion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        prompts: list[str] = []

        async def fake_llm(*, prompt: str, **_kw):
            prompts.append(prompt)
            # La última llamada es la de consolidación
            return _resultado(f"pasada {len(prompts)}")

        monkeypatch.setattr(analyzer_mod, "call_llm_with_schema", fake_llm)
        config = _config(passes=3)

        resultado = await analyze_transcription([_segmento()], config)

        assert len(prompts) == 4, "3 pasadas + 1 consolidación"
        assert resultado is not None and resultado.resumen == "pasada 4"
        assert "consolid" in prompts[-1].lower()

    # Si solo sobrevive una pasada no hay nada que unir: se devuelve tal cual.
    async def test_con_una_sola_pasada_valida_no_consolida(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llamadas = {"n": 0}

        async def fake_llm(**_kw):
            llamadas["n"] += 1
            return _resultado("sobreviviente") if llamadas["n"] == 1 else None

        monkeypatch.setattr(analyzer_mod, "call_llm_with_schema", fake_llm)

        resultado = await analyze_transcription([_segmento()], _config(passes=3))

        assert resultado is not None and resultado.resumen == "sobreviviente"
        assert llamadas["n"] == 3, "no debe haber cuarta llamada de consolidación"

    async def test_si_fallan_todas_devuelve_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_llm(**_kw):
            return None

        monkeypatch.setattr(analyzer_mod, "call_llm_with_schema", fake_llm)

        assert await analyze_transcription([_segmento()], _config(passes=3)) is None

    # Si la consolidación falla, no se pierde el trabajo: cae a la primera pasada.
    async def test_si_falla_la_consolidacion_cae_a_la_primera_pasada(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        llamadas = {"n": 0}

        async def fake_llm(**_kw):
            llamadas["n"] += 1
            return None if llamadas["n"] == 3 else _resultado(f"p{llamadas['n']}")

        monkeypatch.setattr(analyzer_mod, "call_llm_with_schema", fake_llm)

        resultado = await analyze_transcription([_segmento()], _config(passes=2))

        assert resultado is not None and resultado.resumen == "p1"

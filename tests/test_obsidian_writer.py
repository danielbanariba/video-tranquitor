"""Tests para video_tranquitor.writers.obsidian_writer — port de obsidian-writer.test.ts (15 tests)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from video_tranquitor.types import (
    Accionable,
    AnalysisResult,
    AttributedSegment,
    PipelineConfig,
    PipelineResult,
    Requirement,
)
from video_tranquitor.writers.obsidian_writer import write_obsidian_note

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(vault_path: str) -> PipelineConfig:
    return PipelineConfig(
        watch_dir="./Audios",
        output_dir="./output",
        transcriber="openai",
        whisper_cpp_path="",
        whisper_model_path="",
        enable_diarization=False,
        enable_analysis=True,
        enable_obsidian=True,
        enable_toon=False,
        obsidian_vault_path=vault_path,
        hf_token="",
        openai_api_key="sk-test",
        audio_filter="",
        language="es",
        transcription_prompt="",
        transcribe_model="gpt-4o-transcribe",
        whisperx_model="large-v3",
        target_sample_rate=16000,
    )


def make_segment(
    speaker: str | None, text: str, start: float, end: float
) -> AttributedSegment:
    return AttributedSegment(speaker=speaker, text=text, start=start, end=end)


def make_analysis(
    resumen: str = "Resumen de la reunion.",
    requerimientos: list[Requirement] | None = None,
    accionables: list[Accionable] | None = None,
    decisiones: list[str] | None = None,
) -> AnalysisResult:
    if requerimientos is None:
        requerimientos = [
            Requirement(id="REQ-001", descripcion="Login con OAuth", prioridad="alta")
        ]
    if accionables is None:
        accionables = [
            Accionable(responsable="Ana", tarea="Configurar CI/CD", fecha="2026-04-15")
        ]
    if decisiones is None:
        decisiones = ["Migrar a PostgreSQL", "Adoptar TypeScript estricto"]
    return AnalysisResult(
        resumen=resumen,
        requerimientos=requerimientos,
        accionables=accionables,
        decisiones=decisiones,
    )


def make_result(
    input_file: str = "/videos/reunion_semanal.mp4",
    transcription: list[AttributedSegment] | None = None,
    analysis: AnalysisResult | None = None,
    override_analysis: bool = True,
) -> PipelineResult:
    if transcription is None:
        transcription = [
            make_segment("SPEAKER_00", "Buenos dias a todos", 10, 13),
            make_segment("SPEAKER_01", "Hola como estan", 14, 17),
        ]
    if analysis is None and override_analysis:
        analysis = make_analysis()
    return PipelineResult(
        input_file=input_file,
        wav_path="/tmp/reunion_semanal.wav",
        transcription=transcription,
        analysis=analysis,
        toon_output_path=None,
        obsidian_output_path=None,
        duration_ms=1200000,
        audio_duration_sec=1200,
        stages_run=["preprocess", "transcribe", "analyze"],
        whisper_result=None,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------


class TestWriteObsidianNote:
    # 1. Full AnalysisResult → verify YAML frontmatter fields
    def test_generates_markdown_with_all_frontmatter_fields(
        self, tmp_path: Path
    ) -> None:
        result = make_result()
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert re.search(r"^tags:", content, re.MULTILINE)
        assert re.search(r"^fecha:", content, re.MULTILINE)
        assert re.search(r"^estado:", content, re.MULTILINE)
        assert re.search(r"^duracion:", content, re.MULTILINE)
        assert re.search(r"^participantes:", content, re.MULTILINE)

    def test_includes_all_analysis_sections(self, tmp_path: Path) -> None:
        result = make_result()
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "## Resumen" in content
        assert "## Requerimientos" in content
        assert "## Accionables" in content
        assert "## Decisiones" in content

    # 2. No analysis (null) → skips analysis sections
    def test_omits_analysis_sections_when_analysis_is_null(
        self, tmp_path: Path
    ) -> None:
        result = make_result(analysis=None, override_analysis=False)
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "## Transcripción" in content
        assert "## Resumen" not in content
        assert "## Requerimientos" not in content
        assert "## Accionables" not in content
        assert "## Decisiones" not in content

    # 3. Analysis with empty arrays → sections omitted cleanly
    def test_omits_empty_array_sections(self, tmp_path: Path) -> None:
        result = make_result(
            analysis=make_analysis(requerimientos=[], accionables=[], decisiones=[])
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "## Resumen" in content
        assert "## Requerimientos" not in content
        assert "## Accionables" not in content
        assert "## Decisiones" not in content

    # 4. Transcription with speakers → `**SPEAKER_00** (00:00:10): text`
    def test_renders_transcription_with_speaker_attribution(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            transcription=[make_segment("SPEAKER_00", "Buenos dias", 10, 13)]
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "**SPEAKER_00** (00:00:10): Buenos dias" in content

    # 5. Transcription without speakers → `(00:00:10): text`
    def test_renders_transcription_without_speaker_when_null(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            transcription=[make_segment(None, "Texto sin hablante", 10, 13)]
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "(00:00:10): Texto sin hablante" in content
        assert "**None**" not in content
        assert "**null**" not in content

    def test_formats_timestamps_correctly(self, tmp_path: Path) -> None:
        result = make_result(
            transcription=[
                make_segment("SPEAKER_00", "Primero", 0, 2),
                make_segment("SPEAKER_01", "Segundo", 3661, 3665),  # 1h 1m 1s
            ]
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "**SPEAKER_00** (00:00:00): Primero" in content
        assert "**SPEAKER_01** (01:01:01): Segundo" in content

    # 6. Vault path missing → throws E_VAULT_NOT_FOUND
    def test_raises_e_vault_not_found_when_vault_missing(self) -> None:
        result = make_result()
        config = make_config("/absolutely/nonexistent/vault/path/12345")

        with pytest.raises(FileNotFoundError, match="E_VAULT_NOT_FOUND"):
            write_obsidian_note(result, config)

    # Extra: participants extracted from transcription and in frontmatter
    def test_populates_participantes_from_transcription_speakers(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            transcription=[
                make_segment("SPEAKER_00", "Hola", 0, 2),
                make_segment("SPEAKER_01", "Mundo", 3, 5),
            ]
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "SPEAKER_00" in content
        assert "SPEAKER_01" in content

    # Extra: output file created inside vault dir
    def test_writes_file_inside_vault_directory(self, tmp_path: Path) -> None:
        result = make_result()
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)

        assert str(output_path).startswith(str(tmp_path))
        assert output_path.exists()

    # Extra: file name includes date and input base name
    def test_names_output_file_with_date_and_base_name(
        self, tmp_path: Path
    ) -> None:
        result = make_result(input_file="/videos/mi_reunion.mp4")
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        file_name = output_path.name

        assert re.match(r"^\d{4}-\d{2}-\d{2} mi_reunion\.md$", file_name)

    # Extra: requerimientos rendered with checkbox format
    def test_renders_requerimientos_with_checkbox_and_priority(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            analysis=make_analysis(
                requerimientos=[
                    Requirement(id="REQ-001", descripcion="Auth flow", prioridad="alta")
                ]
            )
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "- [ ] **REQ-001** (alta): Auth flow" in content

    # Extra: accionables rendered with responsable and optional fecha
    def test_renders_accionables_with_responsable_tarea_and_fecha(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            analysis=make_analysis(
                accionables=[
                    Accionable(responsable="Juan", tarea="Revisar PR", fecha="2026-04-20")
                ]
            )
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "- [ ] **Juan**: Revisar PR — 2026-04-20" in content

    def test_renders_accionables_without_fecha_when_absent(
        self, tmp_path: Path
    ) -> None:
        result = make_result(
            analysis=make_analysis(
                accionables=[Accionable(responsable="Maria", tarea="Documentar API")]
            )
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "- [ ] **Maria**: Documentar API" in content
        assert "Maria**: Documentar API —" not in content

    # Extra: decisiones rendered as bullet list
    def test_renders_decisiones_as_bullet_list(self, tmp_path: Path) -> None:
        result = make_result(
            analysis=make_analysis(decisiones=["Usar Redis para caché"])
        )
        config = make_config(str(tmp_path))

        output_path = write_obsidian_note(result, config)
        content = output_path.read_text(encoding="utf-8")

        assert "- Usar Redis para caché" in content

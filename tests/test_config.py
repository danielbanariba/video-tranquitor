"""Tests para video_tranquitor.config — port de config.test.ts (14 tests)."""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadConfig:
    """Tests de loadConfig — equivalente al suite de config.test.ts."""

    def _clear_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Elimina todas las variables de entorno relevantes del proceso."""
        vars_to_delete = [
            "OPENAI_API_KEY",
            "TRANSCRIBER",
            "WHISPER_CPP_PATH",
            "WHISPER_MODEL_PATH",
            "ENABLE_DIARIZATION",
            "HF_TOKEN",
            "ENABLE_ANALYSIS",
            "ENABLE_OBSIDIAN",
            "ENABLE_TOON",
            "WATCH_DIR",
            "OUTPUT_DIR",
            "OBSIDIAN_VAULT_PATH",
            "AUDIO_FILTER",
            "LANGUAGE",
            "TRANSCRIPTION_PROMPT",
            "OPENAI_TRANSCRIBE_MODEL",
            "TARGET_SAMPLE_RATE",
        ]
        for var in vars_to_delete:
            monkeypatch.delenv(var, raising=False)

    def _set_minimal_valid_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("TRANSCRIBER", "openai")

    def _load(self) -> object:
        """Importa load_config evitando que dotenv lea .env del disco."""
        # Parcheamos load_dotenv para que no lea archivos .env
        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            return load_config()

    # 1. All defaults
    def test_returns_valid_config_with_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.transcriber == "openai"
        assert cfg.watch_dir == "./Audios"
        assert cfg.output_dir == "./output"
        assert cfg.enable_diarization is False
        assert cfg.enable_analysis is True
        assert cfg.enable_obsidian is True
        assert cfg.enable_toon is True
        assert cfg.language == "es"
        assert cfg.target_sample_rate == 16000
        assert cfg.transcribe_model == "gpt-4o-transcribe"
        assert cfg.openai_api_key == "sk-test-key"

    # 2. TRANSCRIBER=openai + OPENAI_API_KEY missing → raises ValueError
    # (En modo 100% local con TRANSCRIBER=local, la key NO es obligatoria.)
    def test_raises_when_openai_transcriber_and_api_key_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("TRANSCRIBER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            with pytest.raises((ValueError, KeyError), match="OPENAI_API_KEY"):
                load_config()

    # 3. TRANSCRIBER=local + missing WHISPER_CPP_PATH → raises ValueError
    def test_raises_when_transcriber_local_and_whisper_cpp_path_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("TRANSCRIBER", "local")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            with pytest.raises(ValueError, match="WHISPER_CPP_PATH"):
                load_config()

    # 4. TRANSCRIBER=local + missing WHISPER_MODEL_PATH → raises ValueError
    def test_raises_when_transcriber_local_and_whisper_model_path_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("TRANSCRIBER", "local")
        monkeypatch.setenv("WHISPER_CPP_PATH", "/usr/local/bin/whisper")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            with pytest.raises(ValueError, match="WHISPER_MODEL_PATH"):
                load_config()

    # 5. ENABLE_DIARIZATION=true + missing HF_TOKEN → raises ValueError
    def test_raises_when_diarization_enabled_and_hf_token_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("TRANSCRIBER", "openai")
        monkeypatch.setenv("ENABLE_DIARIZATION", "true")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            with pytest.raises(ValueError, match="HF_TOKEN"):
                load_config()

    # 6. All env vars set → fully populated config
    def test_returns_fully_populated_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._clear_env(monkeypatch)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key")
        monkeypatch.setenv("TRANSCRIBER", "local")
        monkeypatch.setenv("WHISPER_CPP_PATH", "/opt/whisper/main")
        monkeypatch.setenv("WHISPER_MODEL_PATH", "/opt/whisper/models/ggml-base.bin")
        monkeypatch.setenv("ENABLE_DIARIZATION", "true")
        monkeypatch.setenv("HF_TOKEN", "hf_abc123")
        monkeypatch.setenv("WATCH_DIR", "/videos")
        monkeypatch.setenv("OUTPUT_DIR", "/out")
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", "/vault/notes")
        monkeypatch.setenv("LANGUAGE", "en")
        monkeypatch.setenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-audio-preview")
        monkeypatch.setenv("TARGET_SAMPLE_RATE", "44100")
        monkeypatch.setenv("ENABLE_ANALYSIS", "true")
        monkeypatch.setenv("ENABLE_OBSIDIAN", "true")
        monkeypatch.setenv("ENABLE_TOON", "true")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.openai_api_key == "sk-real-key"
        assert cfg.transcriber == "local"
        assert cfg.whisper_cpp_path == "/opt/whisper/main"
        assert cfg.whisper_model_path == "/opt/whisper/models/ggml-base.bin"
        assert cfg.enable_diarization is True
        assert cfg.hf_token == "hf_abc123"
        assert cfg.watch_dir == "/videos"
        assert cfg.output_dir == "/out"
        assert cfg.obsidian_vault_path == "/vault/notes"
        assert cfg.language == "en"
        assert cfg.transcribe_model == "gpt-4o-audio-preview"
        assert cfg.target_sample_rate == 44100

    # 7a. ENABLE_ANALYSIS="false" → enable_analysis: False
    def test_sets_enable_analysis_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ENABLE_ANALYSIS", "false")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_analysis is False

    # 7b. ENABLE_ANALYSIS not set → enable_analysis: True
    def test_sets_enable_analysis_true_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.delenv("ENABLE_ANALYSIS", raising=False)

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_analysis is True

    def test_sets_enable_analysis_true_when_explicitly_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ENABLE_ANALYSIS", "true")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_analysis is True

    # 7c. Boolean parsing for ENABLE_OBSIDIAN
    def test_sets_enable_obsidian_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ENABLE_OBSIDIAN", "false")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_obsidian is False

    # 7d. Boolean parsing for ENABLE_TOON
    def test_sets_enable_toon_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ENABLE_TOON", "false")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_toon is False

    # ENABLE_DIARIZATION true/false parsing
    def test_sets_enable_diarization_true_with_hf_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ENABLE_DIARIZATION", "true")
        monkeypatch.setenv("HF_TOKEN", "hf_xyz")

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_diarization is True
        assert cfg.hf_token == "hf_xyz"

    def test_sets_enable_diarization_false_when_not_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)

        from unittest.mock import patch

        with patch("video_tranquitor.config.load_dotenv"):
            from video_tranquitor.config import load_config

            cfg = load_config()

        assert cfg.enable_diarization is False


class TestValidacionDeEsfuerzoPorProveedor:
    """El nivel de esfuerzo se valida contra el proveedor, no contra la unión."""

    # Se reusan los helpers de TestLoadConfig sin heredar de ella: heredar
    # haría que pytest re-corra sus 13 tests dentro de esta clase.
    _set_minimal_valid_env = TestLoadConfig._set_minimal_valid_env
    _clear_env = TestLoadConfig._clear_env
    _load = TestLoadConfig._load

    # Antes se validaba contra la unión de niveles, así que `max` con provider
    # codex pasaba la validación y el CLI lo rechazaba recién en runtime, ya
    # con el audio transcrito y varios minutos gastados.
    def test_max_no_es_valido_para_codex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ANALYSIS_PROVIDER", "codex")
        monkeypatch.setenv("ANALYSIS_EFFORT", "max")

        with pytest.raises(ValueError, match="no es válido para ANALYSIS_PROVIDER=codex"):
            self._load()

    def test_minimal_no_es_valido_para_claude(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ANALYSIS_PROVIDER", "claude")
        monkeypatch.setenv("ANALYSIS_EFFORT", "minimal")

        with pytest.raises(ValueError, match="no es válido para ANALYSIS_PROVIDER=claude"):
            self._load()

    def test_nivel_compartido_sirve_en_ambos(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for provider in ("codex", "claude"):
            self._set_minimal_valid_env(monkeypatch)
            monkeypatch.setenv("ANALYSIS_PROVIDER", provider)
            monkeypatch.setenv("ANALYSIS_EFFORT", "high")

            assert self._load().analysis_effort == "high"

    def test_analysis_passes_tiene_que_ser_al_menos_uno(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("ANALYSIS_PASSES", "0")

        with pytest.raises(ValueError, match="tiene que ser 1 o más"):
            self._load()


class TestVariablesVaciasCaenAlDefault:
    """Una variable definida pero vacía tiene que comportarse como ausente."""

    _clear_env = TestLoadConfig._clear_env
    _set_minimal_valid_env = TestLoadConfig._set_minimal_valid_env
    _load = TestLoadConfig._load

    # LANGUAGE colisiona con la variable estándar de GNU gettext, que muchos
    # sistemas exportan vacía. Una variable definida-pero-vacía no dispara el
    # default de os.environ.get, así que WhisperX recibía language='' y moría.
    def test_language_vacio_cae_al_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("LANGUAGE", "")

        assert self._load().language == "es"

    def test_beam_size_vacio_cae_al_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("WHISPERX_BEAM_SIZE", "")

        assert self._load().whisperx_beam_size == 5

    def test_directorios_vacios_caen_al_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._set_minimal_valid_env(monkeypatch)
        monkeypatch.setenv("WATCH_DIR", "")
        monkeypatch.setenv("OUTPUT_DIR", "")

        cfg = self._load()
        assert cfg.watch_dir == "./Audios"
        assert cfg.output_dir == "./output"

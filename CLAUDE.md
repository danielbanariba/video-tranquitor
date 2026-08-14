# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Video-tranquitor is a Python pipeline (v2.0) that transcribes audio/video files into Spanish, optionally diarizes speakers, runs IA-based analysis to extract requirements/actionables/decisions, and writes the result to a TOON file and/or an Obsidian note. It runs as a one-shot CLI on a single file or as a `watchdog`-based daemon over a drop folder.

The previous TypeScript single-file implementation lives in `.legacy-ts/` for reference. `src/main.ts` no longer exists — the active code is the `video_tranquitor` Python package under `src/`.

## Commands

The Makefile assumes a venv at `./venv` (Python 3.12 — PyTorch does NOT support 3.14):

```bash
make start              # daemon: watch WATCH_DIR (default ./Audios) and process new files
make process FILE=path  # one-shot: process a single file
make test               # pytest tests/ -v
make lint               # ruff check src/ tests/
make format             # ruff format src/ tests/
```

Single test file: `venv/bin/python -m pytest tests/test_config.py -v`
Single test: `venv/bin/python -m pytest tests/test_aligner.py::test_name -v`

The CLI also accepts a positional path: `python -m video_tranquitor path/to/file.mp4`. Without args it falls back to watcher mode.

## Prerequisites

- Python 3.12 venv at `./venv`. **3.14 will not work** — `whisperx` declares `Requires-Python:
  <3.14,>=3.10`, which is why `requires-python` is capped. (Not PyTorch: `torch` declares `>=3.9.0`
  with no upper bound. The older claim that PyTorch was the blocker was wrong.)
  ```
  uv venv --python 3.12 venv
  # torch first, from the CUDA index — the GPU wheels are not on PyPI
  uv pip install --python venv/bin/python torch torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
  uv pip install --python venv/bin/python -e ".[gpu,dev]"
  ```
  Note `uv venv` does **not** seed `pip` into the venv — there is no `venv/bin/pip`, only the
  system-level `pip3`. Always drive installs through `uv pip install --python venv/bin/python`.
  Dependencies are declared as extras because they are genuinely optional:
  - core (always): `pydantic`, `python-dotenv`, `watchdog`, `click` — enough for `TRANSCRIBER=local`
    with diarization off. The analyzer shells out to the `codex`/`claude` CLIs, which are not Python deps.
  - `[gpu]`: `torch`, `torchaudio`, `whisperx`, `pyannote.audio>=4.0`, `numpy` — required for
    `TRANSCRIBER=whisperx|ensemble` and for `ENABLE_DIARIZATION=true`.
  - `[openai]`: only for `TRANSCRIBER=openai`.

  `[tool.uv.sources]` pins `torch`/`torchaudio` to the `cu128` index — the CUDA wheels are not on
  PyPI, so without it you silently get the CPU build and the GPU sits idle. With plain `pip`, install
  them first: `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128`.
- `ffmpeg` and `ffprobe` on PATH (used by `preprocessor.py` and the OpenAI chunker).
- `.env` with `OPENAI_API_KEY=...` — only required when `TRANSCRIBER=openai`; `load_config` does not enforce it otherwise (the `codex`/`claude` CLIs handle their own auth).
- For diarization, accept the user conditions for `pyannote/speaker-diarization-community-1` on Hugging Face and set `HF_TOKEN`. Without an accepted license the pipeline load fails with a 401.
- `codex` CLI on PATH if `ENABLE_ANALYSIS` or `TRANSCRIBER=ensemble` (the analyzer and ensemble arbiter shell out to `codex exec --output-schema`).
- Optional `claude` CLI on PATH — when present, `codex_client` falls back to `claude -p --json-schema` after Codex retries are exhausted (covers quota exhaustion, timeouts, etc.). Uses the user's Claude Code subscription. To opt out, remove `claude` from PATH or rename `src/video_tranquitor/claude_client.py`.

### Env vars consumed by `config.load_config`

Required by transcriber:
- `TRANSCRIBER` — `local` (default) | `openai` | `whisperx` | `ensemble`
- `WHISPER_CPP_PATH`, `WHISPER_MODEL_PATH` — required when transcriber is `local` or `ensemble`
- `HF_TOKEN` — required when `ENABLE_DIARIZATION=true`

Toggles (default ON unless noted):
- `ENABLE_DIARIZATION` (default `false`), `ENABLE_ANALYSIS`, `ENABLE_OBSIDIAN`, `ENABLE_TOON`

Tuning:
- `WATCH_DIR` (default `./Audios`), `OUTPUT_DIR` (default `./output`)
- `OBSIDIAN_VAULT_PATH` (defaults to a hardcoded local path in `config.py` — override in your env)
- `LANGUAGE` (default `es`), `TRANSCRIPTION_PROMPT`, `AUDIO_FILTER`
- `OPENAI_TRANSCRIBE_MODEL` (default `gpt-4o-transcribe`), `WHISPERX_MODEL` (default `large-v3`)
- `DIARIZATION_MODEL` (default `pyannote/speaker-diarization-community-1`)
- `DIARIZATION_EXCLUSIVE` (default `true`) — use the non-overlapping `exclusive_speaker_diarization`
  output when the checkpoint exposes it; falls back to the standard output otherwise
- `ANALYSIS_PROVIDER` — `codex` (default) | `claude`. Picks which CLI runs the analyzer and the
  ensemble arbiter. `codex` keeps its automatic Claude fallback after retries; `claude` goes
  straight to Claude Code and never touches Codex (use it when ChatGPT quota is exhausted)
- `ANALYSIS_MODEL` — model for the selected provider. Empty (default) uses that provider's own
  default. Codex: `gpt-5.6-sol` etc. (or whatever `~/.codex/config.toml` says). Claude: `opus`,
  `sonnet`, `fable`, or a full name like `claude-opus-5`
- `ANALYSIS_EFFORT` — reasoning effort. Codex accepts `minimal|low|medium|high|xhigh`
  (passed as `-c model_reasoning_effort=`); Claude accepts `low|medium|high|xhigh|max`
  (passed as `--effort`). Empty (default) uses the provider default
- `TARGET_SAMPLE_RATE` (default `16000`)

## Architecture

Entry point: `src/video_tranquitor/__main__.py` → `cli.py` (Click). Two modes:
- **One-shot** (`--file` or positional): `asyncio.run(run_pipeline(path, config))`.
- **Watcher** (`--watch` or default): `watcher.start_watcher()` runs a `watchdog.Observer` in a daemon thread, debounces file-creation events (500 ms), pushes paths into a `queue.Queue`, and the asyncio loop drains it and awaits `run_pipeline` per file.

`pipeline.run_pipeline` is the orchestrator. Stages, all guarded by config flags, all timed and reported via `_stage_log`:

1. **Preprocess** (`preprocessor.preprocess_audio`) — ffmpeg → 16kHz mono WAV with `audio_filter` (highpass/lowpass/afftdn). On filter failure it auto-retries without filters. **Do not re-add `loudnorm`**: measured on 49 min of real audio it cost 102 of the 112 seconds of this stage (it resamples internally to 192 kHz) and bought nothing — Whisper already normalizes when computing the log-mel. Dropping it leaves the transcript 94.74% word-identical (differences are punctuation only) and diarization at 97.0% agreement once label permutation is resolved. The remaining filters *do* earn their place: without them the text degrades to 88.61% agreement with visible content errors.
2. **Transcribe** — selected by `config.transcriber`:
   - `openai` → `transcribers/openai_api.py`: chunks audio into 2-min WAV pieces via ffmpeg, calls `client.audio.transcriptions.create()` per chunk, 500 ms sleep between calls. Returns `Transcription[]` directly (no word-level timestamps, so diarization is unavailable).
   - `local` → `transcribers/whispercpp.py`: shells out to `whisper-cli` with `--output-json-full`, parses the JSON into `WhisperResult` (segments + words).
   - `whisperx` → `transcribers/whisperx.py`: loads `whisperx` as a Python lib, transcribes with VAD, then runs forced alignment (wav2vec2) for word-level timestamps. Frees VRAM between stages.
   - `ensemble` → `transcribers/ensemble.py`: runs `whisper.cpp` (in a thread) **and** WhisperX (in a `ProcessPoolExecutor` with `spawn` context, to isolate CUDA state) **in parallel**; both outputs are encoded as TOON and arbitrated by `codex exec` with a JSON schema. Falls back to whichever transcriber succeeded if one fails, or to WhisperX if arbitration fails. Arbitrated chunks have their timestamps validated and snapped against the originals.
3. **Diarize** (`diarizer.diarize`, optional) — only runs when `whisper_result` has word-level tokens. Loads `config.diarization_model` (default `pyannote/speaker-diarization-community-1`, requires pyannote.audio 4.x) with `HF_TOKEN`. Three compatibility shims live here:
   - `_load_pipeline` handles the 3.x→4.x `use_auth_token`→`token` rename.
   - `_load_audio_in_memory` reads the preprocessed WAV with the stdlib `wave` module and hands pyannote a `{"waveform", "sample_rate"}` dict. **This is required, not an optimization** — pyannote 4.x decodes via `torchcodec`, which only supports ffmpeg 4–7. On a system with ffmpeg 8+ the path-based call dies with `NameError: name 'AudioDecoder' is not defined`. Returns `None` for non-PCM WAVs so the caller falls back to passing the path.
   - `_resolve_annotation` handles both output shapes (4.x returns a `DiarizeOutput` with `speaker_diarization` / `exclusive_speaker_diarization`; 3.x returns the `Annotation` directly).
4. **Align** (`aligner.align_speakers`) — binary-search-based assignment of each Whisper word to the diarization segment that best matches its `start` (with a 500 ms tolerance window), then groups consecutive same-speaker words into `AttributedSegment` paragraphs.
5. **Analyze** (`analyzer.analyze_transcription`, optional) — calls `llm_client.call_llm_with_schema`, which dispatches to `codex_client` or `claude_client` per `config.analysis_provider`, to produce `AnalysisResult { resumen, requerimientos[], accionables[], decisiones[], diagrama }`. Failures are non-fatal — pipeline continues.
6. **Write** — `writers/toon_writer.py` (custom inline TOON encoder, byte-compatible with `@toon-format/toon` v2 from the legacy TS) and `writers/obsidian_writer.py` (Markdown with YAML frontmatter, one note per audio in `OBSIDIAN_VAULT_PATH`). Vault-not-found is a soft error.

Cleanup: temp WAVs and chunk dirs are deleted in `finally`/post-stage blocks.

### Data model (`types.py`, all Pydantic v2)

- `WhisperResult { segments: WhisperSegment[], language }` with `WhisperSegment.words: WhisperWord[]` — the canonical intermediate when word-level timestamps exist.
- `Transcription { inicio, fin, texto }` (all strings, `HH:MM:SS`) — chunked output for TOON.
- `AttributedSegment { speaker, text, start, end }` — post-diarization segments.
- `AnalysisResult` with `Requirement` (id/descripcion/prioridad) + `Accionable` (responsable/tarea/fecha, with a validator that normalizes empty/null `fecha` → `None`).
- `PipelineConfig` is the single source of truth for runtime config; never read env vars outside `config.py`.

### Key invariants

- All `Transcription` time fields are `HH:MM:SS` strings; `AttributedSegment` uses float seconds. Conversion happens in `pipeline._time_string_to_seconds`.
- The OpenAI transcriber path **cannot** feed diarization (no `WhisperResult` with words). The pipeline detects this and prints a warning instead of failing.
- The ensemble path uses `multiprocessing.get_context("spawn")` deliberately — CUDA state must NOT be inherited from the parent. Don't switch to `fork`.
- `codex_client` writes the schema to a tempfile, pipes the prompt via stdin, and reads `--output-last-message`. It already has retry-with-backoff (3 attempts, 2s→4s→8s). Don't add another retry layer above it. After all 3 Codex retries fail, it falls back automatically to `claude_client` (which mirrors the same retry pattern via `claude -p --json-schema`) — total worst-case is 3 Codex + 3 Claude attempts before returning `None`.
- `claude_client` is a 1:1 mirror of `codex_client` for Claude Code CLI. Uses `--json-schema` for native structured output, `--tools ""` to disable tool use, `--no-session-persistence` to avoid leaving sessions on disk. Reuses `_extract_json` from `codex_client`. Don't import it directly from analyzer/ensemble — go through `llm_client.call_llm_with_schema`, which is the only provider-selection point.
- `ANALYSIS_MODEL` is never forwarded from Codex to its Claude fallback (an OpenAI model ID is invalid there). `ANALYSIS_EFFORT` is forwarded only when the level exists in both CLIs — see `codex_client.SHARED_EFFORT_LEVELS`. Codex-only `minimal` and Claude-only `max` are dropped on the way across.
- The watcher ignores files starting with `.` or `temp_` — temp WAVs/chunks land in `OUTPUT_DIR` to avoid re-triggering.

## Conventions

- Python 3.12, type hints with `from __future__ import annotations`, `ruff` (line-length 100, rules `E,F,I,UP`).
- `lower_snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, Pydantic `BaseModel` for all data.
- Heavy imports (`torch`, `whisperx`, `pyannote.audio`, `openai`) are **always** lazy-imported inside the function that uses them — startup time and watcher mode depend on this. Don't promote them to module-level.
- All user-facing text (logs, Obsidian notes, error messages) is Spanish. Match the existing tone (sentence-case, no emojis except the `⚠` already in use).
- Commits: short, Spanish, sentence-case (e.g. "Mejorar transcripcion de audio"). Conventional-commit prefixes appear in recent history but the global rule is: no AI attribution / Co-Authored-By.
- Error codes use a `E_*` prefix string at the start of the message (`E_WHISPER_NOT_FOUND`, `E_VAULT_NOT_FOUND`, `E_CODEX_FAILED`, `E_ARBITRATION_FAILED`) — call-sites pattern-match on them for soft-fail behavior.
- Don't commit `temp_*.wav`, `temp_chunks/`, `output/`, `.env`, `venv/`, or anything under `Audios/`.

## Tests

`pytest` with `asyncio_mode = "auto"`. Three test files cover the pieces with the most logic:
- `tests/test_config.py` — env-var parsing and validation in `load_config`
- `tests/test_aligner.py` — speaker alignment edge cases
- `tests/test_obsidian_writer.py` — Markdown/frontmatter rendering

Transcribers, diarizer, and the codex client are not unit-tested — validate them by running the pipeline on a real file in `Audios/`.

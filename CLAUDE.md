# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

Video-tranquitor transcribes video/audio files into Spanish-language transcripts using OpenAI's Whisper API. It processes files from `Audios/`, extracts and preprocesses audio via FFmpeg, chunks it into 2-minute segments for the API, then outputs timestamped transcripts in TOON format (a custom tabular format via `@toon-format/toon`).

## Commands

```bash
npm install   # Install dependencies
npm start     # Run the app (tsx src/main.ts) — processes all files in Audios/
```

No build step needed — `tsx` executes TypeScript directly. No automated tests exist; validate manually by dropping a file in `Audios/` and checking `output/`.

## Prerequisites

- `ffmpeg` and `ffprobe` must be on PATH
- `.env` file with `OPENAI_API_KEY=...`
- Optional env vars: `OPENAI_TRANSCRIBE_MODEL` (default: `gpt-4o-transcribe`), `TRANSCRIPTION_PROMPT`, `AUDIO_FILTER`

## Architecture

Single-file application: `src/main.ts` (~320 lines). Processing pipeline:

1. **Scan** `Audios/` for supported files (`.mp4`, `.mkv`, `.avi`, `.mov`, `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`)
2. **Preprocess** — FFmpeg extracts audio, downsamples to 16kHz mono, applies highpass/lowpass/denoise/loudnorm filters (with fallback if filters fail)
3. **Chunk** — Split into 120-second WAV segments, 500ms delay between API calls for rate limiting
4. **Transcribe** — OpenAI `audio.transcriptions.create()` per chunk, language `es`, returns JSON
5. **Assemble** — Map transcriptions to time ranges (HH:MM:SS), write `{name}_transcription.toon`
6. **Micro-segment** — Split 2-min segments into 10-second chunks, distribute text by sentence boundaries (fallback: word boundaries), write `{name}_detailed.toon`
7. **Cleanup** — Delete temp WAV files and chunk directories

Key data structure: `Transcription { inicio: string, fin: string, texto: string }`

## Conventions

- TypeScript strict mode, ES modules (NodeNext), 2-space indent, double quotes, semicolons
- `lowerCamelCase` for functions/variables, `UPPER_SNAKE_CASE` for constants
- All user-facing text (console logs, output content) in Spanish
- Commits: short, Spanish, sentence-case, no prefixes (e.g. "Mejorar transcripcion de audio")
- Temp artifacts (`temp_*.wav`, `temp_chunks/`) are auto-cleaned and must not be committed

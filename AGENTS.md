# Repository Guidelines

## Project Structure & Module Organization
- `src/main.ts` is the single entry point. It extracts audio, chunks it, and writes `.toon` transcripts.
- `Audios/` is the input drop folder for audio/video files; supported extensions include `.mp4`, `.mkv`, `.avi`, `.mov`, `.ogg`, `.mp3`, `.wav`, `.m4a`, `.flac`.
- `output/` stores generated files like `*_transcription.toon` and `*_detailed.toon`.
- Root-level helpers: `any_tree.py` (directory tree export) and `flac_mp3.py` (local FLAC->MP3 conversion; update paths before use).

## Build, Test, and Development Commands
```
npm install         # install Node dependencies
npm start           # runs tsx src/main.ts; processes files in Audios/
```
Requires `ffmpeg`/`ffprobe` on your PATH and an `.env` file with `OPENAI_API_KEY=...`.

## Coding Style & Naming Conventions
- TypeScript with ES modules (NodeNext) and `strict` enabled. Follow 2-space indentation, double quotes, and semicolons as in `src/main.ts`.
- Use `lowerCamelCase` for functions/variables and `UPPER_SNAKE_CASE` for constants (e.g., `VIDEO_EXTENSIONS`).
- Log/output strings are Spanish; keep new user-facing text in Spanish for consistency.

## Testing Guidelines
- No automated tests are defined. Validate manually:
  1. Place a sample file in `Audios/`.
  2. Run `npm start`.
  3. Confirm outputs in `output/` and spot-check timestamps/text.

## Commit & Pull Request Guidelines
- Commit history uses short, Spanish, sentence-case messages with no prefixes. Keep messages single-line and descriptive (e.g., "Mejorar transcripcion de audio").
- PRs should include: a concise summary, how you tested (command + sample file), and any new dependencies or config changes. If outputs change, mention the generated `.toon` filenames.

## Configuration & Runtime Tips
- Keep secrets in `.env`; never commit API keys.
- Generated files and temp artifacts (`temp_*.mp3`, `temp_*.wav`, `temp_chunks/`) should stay out of commits.

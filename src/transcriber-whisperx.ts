import { spawn } from "child_process";
import { readFileSync, existsSync, unlinkSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { PipelineConfig, WhisperResult, Transcription } from "./types.js";

const WHISPERX_TIMEOUT_MS = 30 * 60 * 1000;

const PROJECT_ROOT = process.cwd();
const VENV_PYTHON = join(PROJECT_ROOT, "venv", "bin", "python");
const WHISPERX_SCRIPT = join(PROJECT_ROOT, "scripts", "transcribe_whisperx.py");

function formatTime(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

export async function transcribeWhisperX(
  audioPath: string,
  config: PipelineConfig,
  modelSize: string = "large-v3"
): Promise<WhisperResult> {
  if (!existsSync(VENV_PYTHON)) {
    throw new Error(`E_VENV_NOT_FOUND: No se encontró Python en ${VENV_PYTHON}`);
  }
  if (!existsSync(WHISPERX_SCRIPT)) {
    throw new Error(`E_SCRIPT_NOT_FOUND: ${WHISPERX_SCRIPT}`);
  }

  const tmpDir = mkdtempSync(join(tmpdir(), "whisperx-"));
  const outputPath = join(tmpDir, "result.json");

  try {
    console.log(`Ejecutando WhisperX (modelo: ${modelSize}, VAD integrado, GPU)...`);

    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        VENV_PYTHON,
        [WHISPERX_SCRIPT, audioPath, modelSize, config.language, outputPath],
        { stdio: ["ignore", "pipe", "pipe"] }
      );

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`WhisperX timeout (${WHISPERX_TIMEOUT_MS}ms)`));
      }, WHISPERX_TIMEOUT_MS);

      child.stderr.on("data", (chunk) => {
        const text = chunk.toString().trim();
        if (text) {
          for (const line of text.split("\n")) {
            if (line.trim()) {
              console.log(`  ${line}`);
            }
          }
        }
      });
      child.stdout.on("data", () => {
        // drain stdout
      });

      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });

      child.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`WhisperX exit ${code}`));
        }
      });
    });

    if (!existsSync(outputPath)) {
      throw new Error("WhisperX no generó archivo de salida");
    }

    const raw = readFileSync(outputPath, "utf-8");
    const parsed = JSON.parse(raw) as WhisperResult;
    return parsed;
  } finally {
    try {
      if (existsSync(outputPath)) unlinkSync(outputPath);
    } catch {
      // ignore
    }
  }
}

export function whisperxResultToTranscriptions(
  result: WhisperResult,
  chunkSeconds: number = 120
): Transcription[] {
  const transcriptions: Transcription[] = [];

  if (result.segments.length === 0) {
    return transcriptions;
  }

  let currentStart = 0;
  let currentEnd = chunkSeconds;
  let currentTexts: string[] = [];

  for (const seg of result.segments) {
    if (seg.start >= currentEnd) {
      if (currentTexts.length > 0) {
        transcriptions.push({
          inicio: formatTime(currentStart),
          fin: formatTime(currentEnd),
          texto: currentTexts.join(" ").trim(),
        });
      }
      currentStart = Math.floor(seg.start / chunkSeconds) * chunkSeconds;
      currentEnd = currentStart + chunkSeconds;
      currentTexts = [];
    }
    currentTexts.push(seg.text.trim());
  }

  if (currentTexts.length > 0) {
    const finalEnd = Math.min(currentEnd, result.segments[result.segments.length - 1].end);
    transcriptions.push({
      inicio: formatTime(currentStart),
      fin: formatTime(finalEnd),
      texto: currentTexts.join(" ").trim(),
    });
  }

  return transcriptions;
}

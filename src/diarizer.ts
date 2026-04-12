import { execFile } from "child_process";
import { promisify } from "util";
import { join } from "path";
import type { PipelineConfig, DiarizationSegment } from "./types.js";

const execFileAsync = promisify(execFile);

const DIARIZATION_TIMEOUT_MS = 30 * 60 * 1000;

/**
 * Run pyannote-audio speaker diarization on the given audio file.
 *
 * Returns an empty array when diarization is disabled or when the HF token
 * is missing (graceful degradation — the pipeline continues without speakers).
 */
export async function diarize(
  audioPath: string,
  config: PipelineConfig
): Promise<DiarizationSegment[]> {
  if (!config.enableDiarization) {
    return [];
  }

  const projectRoot = process.cwd();
  const pythonBin = join(projectRoot, "venv", "bin", "python");
  const scriptPath = join(projectRoot, "scripts", "diarize.py");

  let stdout: string;

  try {
    const result = await execFileAsync(pythonBin, [scriptPath, audioPath], {
      timeout: DIARIZATION_TIMEOUT_MS,
      maxBuffer: 32 * 1024 * 1024, // 32 MB
      env: {
        ...process.env,
        HF_TOKEN: config.hfToken,
      },
    });
    stdout = result.stdout;
  } catch (err) {
    const execErr = err as NodeJS.ErrnoException & { stderr?: string; killed?: boolean };

    if (execErr.killed) {
      throw new Error(
        `El proceso de diarización superó el tiempo límite de 30 minutos para: ${audioPath}`
      );
    }

    const stderr = execErr.stderr ?? "";
    if (stderr.includes("E_HF_TOKEN_MISSING")) {
      console.warn(
        "⚠  Diarización omitida: falta el token de Hugging Face (HF_TOKEN). " +
        "Configurá hfToken en tu .env para habilitar la identificación de hablantes."
      );
      return [];
    }

    throw new Error(
      `Error al ejecutar el proceso de diarización: ${execErr.message}\nStderr: ${stderr}`
    );
  }

  const raw = JSON.parse(stdout.trim()) as Array<{
    speaker: string;
    start: number;
    end: number;
  }>;

  return raw.map((entry) => ({
    speaker: entry.speaker,
    start: entry.start,
    end: entry.end,
  }));
}

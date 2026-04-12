import { execFile, ExecFileException } from "child_process";
import { existsSync, readFileSync, unlinkSync } from "fs";
import { promisify } from "util";
import { formatTime } from "./preprocessor.js";
import type { PipelineConfig, WhisperResult, WhisperSegment, WhisperWord, Transcription } from "./types.js";

const execFileAsync = promisify(execFile);

// 30 minutes in milliseconds
const WHISPER_TIMEOUT_MS = 30 * 60 * 1000;

// Chunk size for grouping segments into Transcription[] (2 minutes in seconds)
const CHUNK_DURATION_SEC = 120;

/**
 * Raw shape of each segment entry in the whisper-cli JSON output.
 * The JSON uses `timestamps.from`/`timestamps.to` as "HH:MM:SS,mmm" strings
 * and `offsets.from`/`offsets.to` as millisecond integers.
 */
interface WhisperJsonSegment {
  timestamps: {
    from: string; // "HH:MM:SS,mmm"
    to: string;
  };
  offsets: {
    from: number; // milliseconds
    to: number;
  };
  text: string;
  tokens?: Array<{
    text: string;
    timestamps?: { from: string; to: string };
    offsets?: { from: number; to: number };
    id: number;
    p: number;
    t_dtw: number;
  }>;
}

interface WhisperJsonOutput {
  result: {
    language: string;
  };
  transcription: WhisperJsonSegment[];
}

/**
 * Parse a whisper timestamp string "HH:MM:SS,mmm" into seconds (float).
 */
function parseWhisperTimestamp(ts: string): number {
  // Format: "HH:MM:SS,mmm"
  const [timePart, msPart] = ts.split(",");
  const [h, m, s] = timePart.split(":").map(Number);
  const ms = msPart ? parseInt(msPart, 10) : 0;
  return h * 3600 + m * 60 + s + ms / 1000;
}

/**
 * Convert the raw whisper-cli JSON output into the internal WhisperResult type.
 */
function parseWhisperJson(jsonPath: string): WhisperResult {
  const raw = readFileSync(jsonPath, "utf-8");
  const data = JSON.parse(raw) as WhisperJsonOutput;

  const segments: WhisperSegment[] = (data.transcription ?? []).map((seg) => {
    const start = seg.offsets.from / 1000;
    const end = seg.offsets.to / 1000;

    const words: WhisperWord[] = (seg.tokens ?? [])
      .filter((tok) => tok.offsets && tok.offsets.from >= 0 && tok.offsets.to >= 0)
      .map((tok) => ({
        word: tok.text,
        start: (tok.offsets!.from) / 1000,
        end: (tok.offsets!.to) / 1000,
      }));

    return {
      text: seg.text.trim(),
      start,
      end,
      words,
    };
  });

  return {
    segments,
    language: data.result?.language ?? "es",
  };
}

/**
 * Invoke whisper-cli to transcribe an audio file using the local whisper.cpp binary.
 *
 * Writes the result JSON alongside the input file (<audioPath>.json) and parses it.
 *
 * @throws Error with code E_WHISPER_NOT_FOUND if the binary does not exist.
 */
export async function transcribeLocal(
  audioPath: string,
  config: PipelineConfig
): Promise<WhisperResult> {
  const binaryPath = config.whisperCppPath;

  if (!existsSync(binaryPath)) {
    const err = new Error(
      `Binario de whisper.cpp no encontrado en: ${binaryPath}. ` +
      "Compilá whisper.cpp con soporte CUDA y configurá WHISPER_CPP_PATH en tu .env"
    );
    (err as NodeJS.ErrnoException).code = "E_WHISPER_NOT_FOUND";
    throw err;
  }

  // whisper-cli writes output to <audioPath>.json when --output-json is used
  // Use --output-file to set explicit output path (without extension) = audioPath itself
  const outputBase = audioPath; // whisper appends ".json"
  const jsonOutputPath = `${audioPath}.json`;

  const args = [
    "-m", config.whisperModelPath,
    "-f", audioPath,
    "-l", config.language,
    "--threads", "4",
    "--output-json-full",
    "--output-file", outputBase,
    "--no-prints",
  ];

  console.log(`Ejecutando whisper.cpp (GPU/CUDA)...`);

  try {
    await execFileAsync(binaryPath, args, {
      timeout: WHISPER_TIMEOUT_MS,
      maxBuffer: 256 * 1024 * 1024, // 256 MB
    });
  } catch (err) {
    const execErr = err as ExecFileException;
    if (execErr.killed) {
      throw new Error(
        `whisper.cpp superó el tiempo límite de 30 minutos para: ${audioPath}`
      );
    }
    throw new Error(
      `Error al ejecutar whisper.cpp: ${execErr.message}\nStderr: ${(execErr as { stderr?: string }).stderr ?? "(sin salida)"}`
    );
  }

  if (!existsSync(jsonOutputPath)) {
    throw new Error(
      `whisper.cpp no generó el archivo JSON esperado en: ${jsonOutputPath}`
    );
  }

  try {
    const result = parseWhisperJson(jsonOutputPath);
    return result;
  } finally {
    // Clean up the JSON file whisper wrote
    if (existsSync(jsonOutputPath)) {
      unlinkSync(jsonOutputPath);
    }
  }
}

/**
 * Convert a WhisperResult (with per-segment timestamps in seconds) to the
 * Transcription[] format used by the TOON writer.
 *
 * Segments are grouped into 2-minute chunks (matching the OpenAI transcriber
 * behavior). Each chunk's text is the concatenation of all segment texts within
 * that window.
 */
export function whisperResultToTranscriptions(result: WhisperResult): Transcription[] {
  if (result.segments.length === 0) {
    return [];
  }

  const transcriptions: Transcription[] = [];
  let chunkStart = result.segments[0].start;
  let chunkEnd = chunkStart + CHUNK_DURATION_SEC;
  let chunkTexts: string[] = [];

  const flush = (actualEnd: number): void => {
    const text = chunkTexts.join(" ").trim();
    if (text) {
      transcriptions.push({
        inicio: formatTime(chunkStart),
        fin: formatTime(actualEnd),
        texto: text,
      });
    }
  };

  for (const seg of result.segments) {
    // If this segment starts past the current chunk boundary, flush and advance
    if (seg.start >= chunkEnd) {
      flush(Math.min(seg.start, chunkEnd));
      chunkStart = chunkEnd;
      // Advance chunk boundaries until we reach the segment
      while (seg.start >= chunkStart + CHUNK_DURATION_SEC) {
        chunkStart += CHUNK_DURATION_SEC;
      }
      chunkEnd = chunkStart + CHUNK_DURATION_SEC;
      chunkTexts = [];
    }
    chunkTexts.push(seg.text);
  }

  // Flush the final chunk
  const lastSeg = result.segments[result.segments.length - 1];
  flush(lastSeg.end);

  return transcriptions;
}

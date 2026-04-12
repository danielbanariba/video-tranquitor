import { config } from "dotenv";
import type { PipelineConfig } from "./types.js";

const DEFAULT_TRANSCRIPTION_PROMPT =
  "Transcribe en espanol con puntuacion clara. Mantiene nombres propios, numeros y siglas tal como se escuchan.";

const DEFAULT_AUDIO_FILTER =
  "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, loudnorm=I=-16:TP=-1.5:LRA=11";

export function loadConfig(): PipelineConfig {
  config();

  const transcriber = (process.env.TRANSCRIBER ?? "local") as "local" | "openai";
  const enableDiarization = process.env.ENABLE_DIARIZATION === "true";

  if (!process.env.OPENAI_API_KEY) {
    throw new Error("OPENAI_API_KEY es requerida. Configurala en tu archivo .env");
  }

  if (transcriber === "local") {
    if (!process.env.WHISPER_CPP_PATH) {
      throw new Error(
        "WHISPER_CPP_PATH es requerida cuando TRANSCRIBER=local. Configurala en tu archivo .env"
      );
    }
    if (!process.env.WHISPER_MODEL_PATH) {
      throw new Error(
        "WHISPER_MODEL_PATH es requerida cuando TRANSCRIBER=local. Configurala en tu archivo .env"
      );
    }
  }

  if (enableDiarization && !process.env.HF_TOKEN) {
    throw new Error(
      "HF_TOKEN es requerida cuando ENABLE_DIARIZATION=true. Configurala en tu archivo .env"
    );
  }

  return {
    watchDir: process.env.WATCH_DIR ?? "./Audios",
    outputDir: process.env.OUTPUT_DIR ?? "./output",
    transcriber,
    whisperCppPath: process.env.WHISPER_CPP_PATH ?? "",
    whisperModelPath: process.env.WHISPER_MODEL_PATH ?? "",
    enableDiarization,
    enableAnalysis: process.env.ENABLE_ANALYSIS !== "false",
    enableObsidian: process.env.ENABLE_OBSIDIAN !== "false",
    enableToon: process.env.ENABLE_TOON !== "false",
    obsidianVaultPath:
      process.env.OBSIDIAN_VAULT_PATH ??
      "/home/banar/Desktop/obsidian/Farinter/07-Reuniones",
    hfToken: process.env.HF_TOKEN ?? "",
    openaiApiKey: process.env.OPENAI_API_KEY,
    audioFilter: process.env.AUDIO_FILTER ?? DEFAULT_AUDIO_FILTER,
    language: process.env.LANGUAGE ?? "es",
    transcriptionPrompt: process.env.TRANSCRIPTION_PROMPT ?? DEFAULT_TRANSCRIPTION_PROMPT,
    transcribeModel: process.env.OPENAI_TRANSCRIBE_MODEL ?? "gpt-4o-transcribe",
    targetSampleRate: process.env.TARGET_SAMPLE_RATE
      ? parseInt(process.env.TARGET_SAMPLE_RATE, 10)
      : 16000,
  };
}

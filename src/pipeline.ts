import { mkdirSync, existsSync, unlinkSync } from "fs";
import { join, basename, extname } from "path";
import { preprocessAudio, getAudioDuration, formatTime } from "./preprocessor.js";
import { transcribeOpenAI } from "./transcriber-openai.js";
import { transcribeLocal, whisperResultToTranscriptions } from "./transcriber.js";
import { transcribeWhisperX, whisperxResultToTranscriptions } from "./transcriber-whisperx.js";
import { transcribeEnsemble } from "./transcriber-ensemble.js";
import { diarize } from "./diarizer.js";
import { alignSpeakers } from "./aligner.js";
import { writeToon } from "./writer/toon-writer.js";
import { writeObsidianNote } from "./writer/obsidian-writer.js";
import { analyzeTranscription } from "./analyzer.js";
import type { PipelineConfig, PipelineResult, AttributedSegment, WhisperResult, AnalysisResult } from "./types.js";

export const VIDEO_EXTENSIONS = new Set([".mp4", ".mkv", ".avi", ".mov"]);
export const AUDIO_EXTENSIONS = new Set([".ogg", ".mp3", ".wav", ".m4a", ".flac"]);

function stageLog(label: string, startMs: number): void {
  const elapsed = ((Date.now() - startMs) / 1000).toFixed(2);
  console.log(`  [${label}] completado en ${elapsed}s`);
}

export async function runPipeline(
  filePath: string,
  config: PipelineConfig
): Promise<PipelineResult> {
  const pipelineStart = Date.now();
  const stagesRun: string[] = [];

  const baseName = basename(filePath, extname(filePath));
  const ext = extname(filePath).toLowerCase();
  const isVideo = VIDEO_EXTENSIONS.has(ext);

  const tempWavPath = join(config.outputDir, `temp_${baseName}.wav`);

  if (!existsSync(config.outputDir)) {
    mkdirSync(config.outputDir, { recursive: true });
  }

  console.log(`\nIniciando pipeline para: ${baseName}`);

  // Etapa 1: Preprocesamiento de audio
  let stageStart = Date.now();
  console.log(isVideo ? "Extrayendo y optimizando audio del video..." : "Optimizando audio...");

  const preprocessOk = preprocessAudio(
    filePath,
    tempWavPath,
    config.audioFilter,
    config.targetSampleRate
  );

  if (!preprocessOk) {
    throw new Error(`No se pudo preprocesar el archivo: ${filePath}`);
  }

  stagesRun.push("preprocess");
  stageLog("preprocess", stageStart);

  const duration = getAudioDuration(tempWavPath);
  console.log(`Duración total del audio: ${formatTime(duration)}`);

  // Etapa 2: Transcripción
  stageStart = Date.now();
  console.log("Transcribiendo audio...");

  let rawTranscriptions;
  let whisperResult: WhisperResult | null = null;

  if (config.transcriber === "openai") {
    rawTranscriptions = await transcribeOpenAI(
      tempWavPath,
      config.openaiApiKey,
      config.transcribeModel,
      config.transcriptionPrompt,
      config.targetSampleRate,
      config.language
    );
  } else if (config.transcriber === "whisperx") {
    whisperResult = await transcribeWhisperX(tempWavPath, config, config.whisperxModel);
    rawTranscriptions = whisperxResultToTranscriptions(whisperResult);
  } else if (config.transcriber === "ensemble") {
    const ensembleResult = await transcribeEnsemble(tempWavPath, config);
    whisperResult = ensembleResult.whisperResult;
    rawTranscriptions = ensembleResult.arbitrated;
  } else {
    // config.transcriber === "local"
    whisperResult = await transcribeLocal(tempWavPath, config);
    rawTranscriptions = whisperResultToTranscriptions(whisperResult);
  }

  stagesRun.push("transcribe");
  stageLog("transcribe", stageStart);

  // Mapear Transcription[] -> AttributedSegment[] (base: sin hablantes)
  let transcription: AttributedSegment[] = rawTranscriptions.map((t) => ({
    speaker: null,
    text: t.texto,
    start: timeStringToSeconds(t.inicio),
    end: timeStringToSeconds(t.fin),
  }));

  // Etapa 3: Diarización
  if (config.enableDiarization) {
    if (
      whisperResult === null ||
      whisperResult.segments.length === 0 ||
      whisperResult.segments[0]?.words?.length === 0
    ) {
      console.warn(
        "⚠  Diarización requiere timestamps a nivel de palabra. Omitiendo diarización."
      );
    } else if (whisperResult !== null) {
      stageStart = Date.now();
      console.log("Ejecutando diarización de hablantes...");

      const diarizationSegments = await diarize(tempWavPath, config);

      if (diarizationSegments.length > 0) {
        transcription = alignSpeakers(whisperResult, diarizationSegments);
      }

      stagesRun.push("diarization");
      stageLog("diarization", stageStart);
    }
  }

  // Etapa 4: Análisis
  let analysis: AnalysisResult | null = null;
  if (config.enableAnalysis) {
    stageStart = Date.now();
    console.log("Analizando transcripción con IA...");
    analysis = await analyzeTranscription(transcription, config);
    if (analysis) {
      stagesRun.push("analysis");
      stageLog("analysis", stageStart);
    } else {
      console.warn("  Análisis omitido (falló o no disponible). El pipeline continúa.");
    }
  }

  // Etapa 5: Escritura TOON
  let toonOutputPath: string | null = null;
  if (config.enableToon) {
    stageStart = Date.now();
    const toonPath = join(config.outputDir, `${baseName}_transcription.toon`);
    writeToon(rawTranscriptions, toonPath);
    toonOutputPath = toonPath;
    stagesRun.push("toon");
    stageLog("toon", stageStart);
    console.log(`Archivo TOON guardado en: ${toonPath}`);
  }

  // Etapa 6: Obsidian
  let obsidianOutputPath: string | null = null;
  if (config.enableObsidian) {
    stageStart = Date.now();
    console.log("Generando nota en Obsidian...");
    try {
      const partialResult: PipelineResult = {
        inputFile: filePath,
        wavPath: tempWavPath,
        transcription,
        analysis,
        toonOutputPath,
        obsidianOutputPath: null,
        durationMs: Date.now() - pipelineStart,
        audioDurationSec: duration,
        stagesRun,
        whisperResult,
      };
      obsidianOutputPath = writeObsidianNote(partialResult, config);
      stagesRun.push("obsidian");
      stageLog("obsidian", stageStart);
      console.log(`Nota de Obsidian guardada en: ${obsidianOutputPath}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      if (message.startsWith("E_VAULT_NOT_FOUND")) {
        console.error(`  ${message}`);
        console.warn("  Nota de Obsidian omitida. El pipeline continúa.");
      } else {
        console.error("  Error al escribir nota de Obsidian:", message);
        console.warn("  Nota de Obsidian omitida. El pipeline continúa.");
      }
    }
  }

  // Limpieza de archivos temporales
  if (existsSync(tempWavPath)) {
    unlinkSync(tempWavPath);
  }

  const durationMs = Date.now() - pipelineStart;
  console.log(`\nPipeline finalizado para: ${baseName} (${(durationMs / 1000).toFixed(2)}s total)`);

  return {
    inputFile: filePath,
    wavPath: tempWavPath,
    transcription,
    analysis,
    toonOutputPath,
    obsidianOutputPath,
    durationMs,
    audioDurationSec: duration,
    stagesRun,
    whisperResult,
  };
}

function timeStringToSeconds(time: string): number {
  const [h, m, s] = time.split(":").map(Number);
  return h * 3600 + m * 60 + s;
}

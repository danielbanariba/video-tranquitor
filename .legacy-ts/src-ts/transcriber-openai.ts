import { execSync } from "child_process";
import { readFileSync, mkdirSync, existsSync, unlinkSync, rmSync } from "fs";
import { join, basename, extname } from "path";
import OpenAI from "openai";
import { getAudioDuration, getFileSizeMB, formatTime } from "./preprocessor.js";
import type { Transcription } from "./types.js";

async function transcribeWithOpenAI(
  audioPath: string,
  client: OpenAI,
  transcribeModel: string,
  transcriptionPrompt: string,
  language: string = "es"
): Promise<string> {
  try {
    const fileSize = getFileSizeMB(audioPath);
    if (fileSize > 24) {
      console.log(`Advertencia: El archivo es de ${fileSize.toFixed(2)} MB`);
    }

    const audioFile = readFileSync(audioPath);
    const fileType = extname(audioPath).toLowerCase() === ".wav" ? "audio/wav" : "audio/mpeg";
    const file = new File([audioFile], basename(audioPath), { type: fileType });

    const prompt = transcriptionPrompt.trim();
    const response = await client.audio.transcriptions.create({
      model: transcribeModel,
      file: file,
      language: language,
      response_format: "json",
      ...(prompt ? { prompt } : {}),
    });

    return response.text.trim();
  } catch (error) {
    console.error("Error en la transcripcion con OpenAI:", error);
    return "";
  }
}

async function transcribeLongAudio(
  audioPath: string,
  client: OpenAI,
  transcribeModel: string,
  transcriptionPrompt: string,
  targetSampleRate: number,
  language: string = "es",
  chunkLengthSec: number = 120
): Promise<Transcription[]> {
  const duration = getAudioDuration(audioPath);
  const durationMs = duration * 1000;
  const chunkLengthMs = chunkLengthSec * 1000;
  const transcriptions: Transcription[] = [];

  const tempDir = "temp_chunks";
  if (!existsSync(tempDir)) {
    mkdirSync(tempDir);
  }

  const totalChunks = Math.ceil(durationMs / chunkLengthMs);

  for (let i = 0; i < durationMs; i += chunkLengthMs) {
    const chunkIndex = Math.floor(i / chunkLengthMs) + 1;
    const startSec = i / 1000;
    const durationSec = Math.min(chunkLengthSec, (durationMs - i) / 1000);
    const chunkFilename = join(tempDir, `chunk_${i}.wav`);

    try {
      execSync(
        `ffmpeg -i "${audioPath}" -ss ${startSec} -t ${durationSec} -ac 1 -ar ${targetSampleRate} -sample_fmt s16 -c:a pcm_s16le -y "${chunkFilename}"`,
        { stdio: "pipe" }
      );
    } catch {
      console.error(`Error al crear chunk ${chunkIndex}`);
      continue;
    }

    const fileSize = getFileSizeMB(chunkFilename);
    console.log(`Transcribiendo segmento ${chunkIndex}/${totalChunks} (Tamano: ${fileSize.toFixed(2)} MB)`);

    const text = await transcribeWithOpenAI(chunkFilename, client, transcribeModel, transcriptionPrompt, language);

    if (text) {
      const startTime = i / 1000;
      const endTime = Math.min((i + chunkLengthMs) / 1000, duration);

      transcriptions.push({
        inicio: formatTime(startTime),
        fin: formatTime(endTime),
        texto: text,
      });
    }

    unlinkSync(chunkFilename);
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  rmSync(tempDir, { recursive: true, force: true });
  return transcriptions;
}

export function divideSegmentsInSmallerChunks(
  transcriptions: Transcription[],
  smallChunkSeconds: number = 10
): Transcription[] {
  const smallSegments: Transcription[] = [];

  for (const segment of transcriptions) {
    const [startH, startM, startS] = segment.inicio.split(":").map(Number);
    const startSeconds = startH * 3600 + startM * 60 + startS;

    const [endH, endM, endS] = segment.fin.split(":").map(Number);
    const endSeconds = endH * 3600 + endM * 60 + endS;

    const duration = endSeconds - startSeconds;

    if (duration <= smallChunkSeconds) {
      smallSegments.push(segment);
      continue;
    }

    const texto = segment.texto;
    const numSmallChunks = Math.ceil(duration / smallChunkSeconds);

    const sentences: string[] = [];
    let currentSentence = "";
    for (const char of texto) {
      currentSentence += char;
      if (".!?".includes(char)) {
        sentences.push(currentSentence.trim());
        currentSentence = "";
      }
    }
    if (currentSentence) {
      sentences.push(currentSentence.trim());
    }

    let finalSentences = sentences;
    if (sentences.length < numSmallChunks / 2) {
      const words = texto.split(/\s+/);
      const approxWordsPerChunk = Math.max(1, Math.floor(words.length / numSmallChunks));
      finalSentences = [];
      for (let i = 0; i < words.length; i += approxWordsPerChunk) {
        finalSentences.push(words.slice(i, i + approxWordsPerChunk).join(" "));
      }
    }

    for (let i = 0; i < numSmallChunks; i++) {
      const smallStart = startSeconds + i * smallChunkSeconds;
      const smallEnd = Math.min(smallStart + smallChunkSeconds, endSeconds);

      const sentenceStartIdx = Math.floor((i / numSmallChunks) * finalSentences.length);
      let sentenceEndIdx = Math.floor(((i + 1) / numSmallChunks) * finalSentences.length);

      if (sentenceStartIdx === sentenceEndIdx && sentenceStartIdx < finalSentences.length) {
        sentenceEndIdx = sentenceStartIdx + 1;
      }

      let segmentText = finalSentences.slice(sentenceStartIdx, sentenceEndIdx).join(" ").trim();
      if (!segmentText) {
        segmentText = "[silencio o ruido de fondo]";
      }

      smallSegments.push({
        inicio: formatTime(smallStart),
        fin: formatTime(smallEnd),
        texto: segmentText,
      });
    }
  }

  return smallSegments;
}

export async function transcribeOpenAI(
  audioPath: string,
  openaiApiKey: string,
  transcribeModel: string,
  transcriptionPrompt: string,
  targetSampleRate: number,
  language: string = "es"
): Promise<Transcription[]> {
  const client = new OpenAI({ apiKey: openaiApiKey });
  return transcribeLongAudio(
    audioPath,
    client,
    transcribeModel,
    transcriptionPrompt,
    targetSampleRate,
    language,
    120
  );
}

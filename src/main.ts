import { execSync } from "child_process";
import { readFileSync, writeFileSync, mkdirSync, readdirSync, unlinkSync, existsSync, statSync, rmSync } from "fs";
import { join, basename, extname } from "path";
import { config } from "dotenv";
import OpenAI from "openai";
import { encode } from "@toon-format/toon";

config();

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const TRANSCRIBE_MODEL = process.env.OPENAI_TRANSCRIBE_MODEL ?? "gpt-4o-transcribe";
const TRANSCRIPTION_PROMPT =
  process.env.TRANSCRIPTION_PROMPT ??
  "Transcribe en espanol con puntuacion clara. Mantiene nombres propios, numeros y siglas tal como se escuchan.";
const AUDIO_FILTER =
  process.env.AUDIO_FILTER ??
  "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, loudnorm=I=-16:TP=-1.5:LRA=11";
const TARGET_SAMPLE_RATE = 16000;

interface Transcription {
  inicio: string;
  fin: string;
  texto: string;
}

function preprocessAudio(inputPath: string, outputPath: string): boolean {
  const baseCommand = `ffmpeg -i "${inputPath}" -vn -ac 1 -ar ${TARGET_SAMPLE_RATE} -sample_fmt s16`;
  const filterArg = AUDIO_FILTER ? ` -af "${AUDIO_FILTER}"` : "";
  try {
    execSync(
      `${baseCommand}${filterArg} -y "${outputPath}"`,
      {
        stdio: "pipe",
      }
    );
    return true;
  } catch (error) {
    if (filterArg) {
      console.error("Error al preparar el audio con filtros, reintentando sin filtros:", error);
      try {
        execSync(`${baseCommand} -y "${outputPath}"`, { stdio: "pipe" });
        return true;
      } catch (fallbackError) {
        console.error("Error al preparar el audio:", fallbackError);
        return false;
      }
    }
    console.error("Error al preparar el audio:", error);
    return false;
  }
}

function getAudioDuration(audioPath: string): number {
  try {
    const result = execSync(
      `ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${audioPath}"`,
      { encoding: "utf-8" }
    );
    return parseFloat(result.trim());
  } catch {
    return 0;
  }
}

function getFileSizeMB(filePath: string): number {
  return statSync(filePath).size / (1024 * 1024);
}

function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

async function transcribeWithOpenAI(audioPath: string, language: string = "es"): Promise<string> {
  try {
    const fileSize = getFileSizeMB(audioPath);
    if (fileSize > 24) {
      console.log(`Advertencia: El archivo es de ${fileSize.toFixed(2)} MB`);
    }

    const audioFile = readFileSync(audioPath);
    const fileType = extname(audioPath).toLowerCase() === ".wav" ? "audio/wav" : "audio/mpeg";
    const file = new File([audioFile], basename(audioPath), { type: fileType });

    const prompt = TRANSCRIPTION_PROMPT.trim();
    const response = await client.audio.transcriptions.create({
      model: TRANSCRIBE_MODEL,
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
        `ffmpeg -i "${audioPath}" -ss ${startSec} -t ${durationSec} -ac 1 -ar ${TARGET_SAMPLE_RATE} -sample_fmt s16 -c:a pcm_s16le -y "${chunkFilename}"`,
        { stdio: "pipe" }
      );
    } catch {
      console.error(`Error al crear chunk ${chunkIndex}`);
      continue;
    }

    const fileSize = getFileSizeMB(chunkFilename);
    console.log(`Transcribiendo segmento ${chunkIndex}/${totalChunks} (Tamano: ${fileSize.toFixed(2)} MB)`);

    const text = await transcribeWithOpenAI(chunkFilename, language);

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

function divideSegmentsInSmallerChunks(
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

function saveToFile(data: Transcription[], outputPath: string): void {
  const toonContent = encode({ transcripciones: data });
  writeFileSync(outputPath, toonContent, "utf-8");
}

async function main(filePath: string, isVideo: boolean = true, language: string = "es"): Promise<boolean> {
  if (!process.env.OPENAI_API_KEY) {
    console.error("Error: No se ha configurado la API key de OpenAI en el archivo .env");
    console.log("Por favor, crea un archivo .env con: OPENAI_API_KEY=tu_api_key_aqui");
    return false;
  }

  const baseName = basename(filePath, extname(filePath));
  const tempWavPath = `temp_${baseName}.wav`;

  const outputDir = "output";
  if (!existsSync(outputDir)) {
    mkdirSync(outputDir);
  }

  const outputTextPath = join(outputDir, `${baseName}_transcription.toon`);
  const outputDetailedPath = join(outputDir, `${baseName}_detailed.toon`);

  try {
    console.log(`Procesando archivo: ${baseName}`);

    if (isVideo) {
      console.log("Extrayendo y optimizando audio del video...");
      if (!preprocessAudio(filePath, tempWavPath)) {
        return false;
      }
    } else {
      console.log("Optimizando audio...");
      if (!preprocessAudio(filePath, tempWavPath)) {
        return false;
      }
    }

    const duration = getAudioDuration(tempWavPath);
    console.log(`Duracion total del audio: ${formatTime(duration)}`);

    console.log("Transcribiendo audio en segmentos de 2 minutos...");
    const transcriptions = await transcribeLongAudio(tempWavPath, language, 120);

    saveToFile(transcriptions, outputTextPath);
    console.log(`Transcripcion completada para: ${baseName}`);
    console.log(`Archivo guardado en: ${outputTextPath}`);

    console.log("Dividiendo transcripcion en segmentos de 10 segundos...");
    const smallSegments = divideSegmentsInSmallerChunks(transcriptions, 10);

    saveToFile(smallSegments, outputDetailedPath);
    console.log(`Transcripcion detallada guardada en: ${outputDetailedPath}\n`);

    return true;
  } catch (error) {
    console.error(`Error durante el procesamiento de ${baseName}:`, error);
    return false;
  } finally {
    if (existsSync(tempWavPath)) {
      unlinkSync(tempWavPath);
    }
  }
}

const VIDEO_EXTENSIONS = new Set([".mp4", ".mkv", ".avi", ".mov"]);
const AUDIO_EXTENSIONS = new Set([".ogg", ".mp3", ".wav", ".m4a", ".flac"]);

const folderPath = "./Audios";
const language = "es";

if (existsSync(folderPath)) {
  const files = readdirSync(folderPath);
  for (const filename of files) {
    const ext = extname(filename).toLowerCase();
    const filePath = join(folderPath, filename);

    if (VIDEO_EXTENSIONS.has(ext)) {
      await main(filePath, true, language);
    } else if (AUDIO_EXTENSIONS.has(ext)) {
      await main(filePath, false, language);
    }
  }
} else {
  console.log(`La carpeta ${folderPath} no existe. Creala y agrega archivos de audio/video.`);
  mkdirSync(folderPath, { recursive: true });
}

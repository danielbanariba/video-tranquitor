import { mkdirSync, readdirSync, existsSync, unlinkSync } from "fs";
import { join, basename, extname } from "path";
import { config } from "dotenv";
import { preprocessAudio, getAudioDuration, formatTime } from "./preprocessor.js";
import { transcribeOpenAI, divideSegmentsInSmallerChunks } from "./transcriber-openai.js";
import { writeToon } from "./writer/toon-writer.js";

config();

const TRANSCRIBE_MODEL = process.env.OPENAI_TRANSCRIBE_MODEL ?? "gpt-4o-transcribe";
const TRANSCRIPTION_PROMPT =
  process.env.TRANSCRIPTION_PROMPT ??
  "Transcribe en espanol con puntuacion clara. Mantiene nombres propios, numeros y siglas tal como se escuchan.";
const AUDIO_FILTER =
  process.env.AUDIO_FILTER ??
  "highpass=f=80, lowpass=f=12000, afftdn=nf=-25, loudnorm=I=-16:TP=-1.5:LRA=11";
const TARGET_SAMPLE_RATE = 16000;

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
    } else {
      console.log("Optimizando audio...");
    }

    if (!preprocessAudio(filePath, tempWavPath, AUDIO_FILTER, TARGET_SAMPLE_RATE)) {
      return false;
    }

    const duration = getAudioDuration(tempWavPath);
    console.log(`Duracion total del audio: ${formatTime(duration)}`);

    console.log("Transcribiendo audio en segmentos de 2 minutos...");
    const transcriptions = await transcribeOpenAI(
      tempWavPath,
      process.env.OPENAI_API_KEY,
      TRANSCRIBE_MODEL,
      TRANSCRIPTION_PROMPT,
      TARGET_SAMPLE_RATE,
      language
    );

    writeToon(transcriptions, outputTextPath);
    console.log(`Transcripcion completada para: ${baseName}`);
    console.log(`Archivo guardado en: ${outputTextPath}`);

    console.log("Dividiendo transcripcion en segmentos de 10 segundos...");
    const smallSegments = divideSegmentsInSmallerChunks(transcriptions, 10);

    writeToon(smallSegments, outputDetailedPath);
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

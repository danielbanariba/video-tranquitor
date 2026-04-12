import { spawn } from "child_process";
import { writeFileSync, readFileSync, existsSync, unlinkSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import type { AttributedSegment, AnalysisResult, PipelineConfig } from "./types.js";

const MAX_RETRIES = 3;
const INITIAL_BACKOFF_MS = 2000;
const CODEX_TIMEOUT_MS = 10 * 60 * 1000;

const ANALYSIS_SCHEMA = {
  type: "object",
  properties: {
    resumen: { type: "string" },
    requerimientos: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          descripcion: { type: "string" },
          prioridad: { type: "string", enum: ["alta", "media", "baja"] },
        },
        required: ["id", "descripcion", "prioridad"],
        additionalProperties: false,
      },
    },
    accionables: {
      type: "array",
      items: {
        type: "object",
        properties: {
          responsable: { type: "string" },
          tarea: { type: "string" },
          fecha: { type: ["string", "null"] },
        },
        required: ["responsable", "tarea", "fecha"],
        additionalProperties: false,
      },
    },
    decisiones: {
      type: "array",
      items: { type: "string" },
    },
  },
  required: ["resumen", "requerimientos", "accionables", "decisiones"],
  additionalProperties: false,
};

function buildTranscriptionText(transcription: AttributedSegment[]): string {
  return transcription
    .map((seg) => {
      const timeLabel = formatSeconds(seg.start);
      if (seg.speaker) {
        return `${seg.speaker} (${timeLabel}): ${seg.text}`;
      }
      return `(${timeLabel}): ${seg.text}`;
    })
    .join("\n");
}

function formatSeconds(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function buildPrompt(transcriptionText: string): string {
  return `Eres un experto en ingeniería de requerimientos. Analiza la siguiente transcripción de una reunión y extrae información estructurada.

Transcripción:
${transcriptionText}

Devuelve un JSON válido con esta estructura:
- resumen: Resumen ejecutivo de la reunión en 1-2 párrafos
- requerimientos: array con id ("REQ-001"), descripcion, prioridad ("alta"|"media"|"baja")
- accionables: array con responsable, tarea, fecha opcional (YYYY-MM-DD)
- decisiones: array de strings

Si no hay requerimientos, accionables o decisiones, devuelve arrays vacíos.`;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function callCodex(prompt: string): Promise<string> {
  const tmpDir = mkdtempSync(join(tmpdir(), "codex-analyzer-"));
  const schemaPath = join(tmpDir, "schema.json");
  const outputPath = join(tmpDir, "output.txt");

  writeFileSync(schemaPath, JSON.stringify(ANALYSIS_SCHEMA), "utf-8");

  try {
    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        "codex",
        [
          "exec",
          "--sandbox", "read-only",
          "--skip-git-repo-check",
          "--ephemeral",
          "--output-schema", schemaPath,
          "--output-last-message", outputPath,
          "--color", "never",
          "-",
        ],
        { stdio: ["pipe", "pipe", "pipe"] }
      );

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`Codex timeout (${CODEX_TIMEOUT_MS}ms)`));
      }, CODEX_TIMEOUT_MS);

      let stderr = "";
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.stdout.on("data", () => {
        // drain stdout to prevent buffer blocking
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
          reject(new Error(`Codex exit ${code}: ${stderr.slice(-500)}`));
        }
      });

      child.stdin.write(prompt);
      child.stdin.end();
    });

    if (!existsSync(outputPath)) {
      throw new Error("Codex no generó archivo de salida");
    }

    const content = readFileSync(outputPath, "utf-8").trim();
    if (!content) {
      throw new Error("Respuesta vacía de Codex");
    }

    return content;
  } finally {
    try {
      if (existsSync(schemaPath)) unlinkSync(schemaPath);
      if (existsSync(outputPath)) unlinkSync(outputPath);
    } catch {
      // ignore cleanup errors
    }
  }
}

function extractJson(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }

  return trimmed;
}

export async function analyzeTranscription(
  transcription: AttributedSegment[],
  _config: PipelineConfig
): Promise<AnalysisResult | null> {
  const transcriptionText = buildTranscriptionText(transcription);
  const prompt = buildPrompt(transcriptionText);

  let lastError: unknown = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const rawOutput = await callCodex(prompt);
      const jsonText = extractJson(rawOutput);
      const parsed = JSON.parse(jsonText) as AnalysisResult;

      if (
        typeof parsed.resumen !== "string" ||
        !Array.isArray(parsed.requerimientos) ||
        !Array.isArray(parsed.accionables) ||
        !Array.isArray(parsed.decisiones)
      ) {
        throw new Error("Estructura JSON inválida en la respuesta de Codex");
      }

      parsed.accionables = parsed.accionables.map((a) => ({
        responsable: a.responsable,
        tarea: a.tarea,
        ...(a.fecha ? { fecha: a.fecha } : {}),
      }));

      return parsed;
    } catch (error) {
      lastError = error;

      if (attempt < MAX_RETRIES) {
        const backoffMs = INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
        console.warn(
          `  Análisis: intento ${attempt}/${MAX_RETRIES} fallido. Reintentando en ${backoffMs / 1000}s...`
        );
        await sleep(backoffMs);
      }
    }
  }

  console.error(
    `E_CODEX_FAILED: Análisis fallido tras ${MAX_RETRIES} intentos.`,
    lastError instanceof Error ? lastError.message : lastError
  );

  return null;
}

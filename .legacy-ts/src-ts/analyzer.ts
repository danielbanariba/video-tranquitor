import type { AttributedSegment, AnalysisResult, PipelineConfig } from "./types.js";
import { callCodexWithSchema } from "./codex-client.js";

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

function validateAnalysisResult(parsed: unknown): AnalysisResult {
  const p = parsed as AnalysisResult;
  if (
    typeof p.resumen !== "string" ||
    !Array.isArray(p.requerimientos) ||
    !Array.isArray(p.accionables) ||
    !Array.isArray(p.decisiones)
  ) {
    throw new Error("Estructura JSON inválida en la respuesta de Codex");
  }

  p.accionables = p.accionables.map((a) => ({
    responsable: a.responsable,
    tarea: a.tarea,
    ...(a.fecha ? { fecha: a.fecha } : {}),
  }));

  return p;
}

export async function analyzeTranscription(
  transcription: AttributedSegment[],
  _config: PipelineConfig
): Promise<AnalysisResult | null> {
  const transcriptionText = buildTranscriptionText(transcription);
  const prompt = buildPrompt(transcriptionText);

  return callCodexWithSchema(prompt, ANALYSIS_SCHEMA, validateAnalysisResult, {
    maxRetries: 3,
    timeoutMs: 10 * 60 * 1000,
    errorCode: "E_CODEX_FAILED",
  });
}

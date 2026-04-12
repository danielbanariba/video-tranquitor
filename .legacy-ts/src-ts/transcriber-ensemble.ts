import { encode } from "@toon-format/toon";
import { transcribeLocal, whisperResultToTranscriptions } from "./transcriber.js";
import { transcribeWhisperX, whisperxResultToTranscriptions } from "./transcriber-whisperx.js";
import { callCodexWithSchema } from "./codex-client.js";
import type { PipelineConfig, EnsembleResult, Transcription } from "./types.js";

const ARBITRATION_SCHEMA = {
  type: "object",
  properties: {
    transcripciones: {
      type: "array",
      items: {
        type: "object",
        properties: {
          inicio: { type: "string" },
          fin: { type: "string" },
          texto: { type: "string" },
        },
        required: ["inicio", "fin", "texto"],
        additionalProperties: false,
      },
    },
  },
  required: ["transcripciones"],
  additionalProperties: false,
};

function buildArbitrationPrompt(toonTurbo: string, toonWhisperx: string): string {
  return `Sos un experto en transcripción de audio en español. Te paso DOS transcripciones del mismo audio en formato TOON (tabular), una de whisper.cpp turbo y otra de WhisperX con large-v3.

Tu tarea: producir UNA transcripción consensuada, chunk por chunk, eligiendo la mejor versión basándote en el CONTEXTO global de la reunión.

Reglas:
- Cada chunk tiene los mismos timestamps (inicio, fin) en ambas transcripciones — conservalos EXACTOS
- Cuando difieran en un nombre propio, elegí el que sea consistente con el resto de la reunión (ej: si "Edwin" aparece 10 veces y "Edo" solo 1, probablemente era "Edwin" en todos los casos)
- Cuando difieran en términos técnicos, elegí el que tenga sentido en el contexto
- Cuando una versión tenga duplicaciones (mismo texto en 2 chunks seguidos), preferí la otra
- Si no podés decidir con certeza, usá la versión de WhisperX (es más precisa en general)
- NO inventes texto nuevo: solo elegí entre las dos versiones existentes, o combinalas palabra por palabra
- NO cambies los timestamps

Devolvé el resultado como JSON con la estructura: { "transcripciones": [{inicio, fin, texto}] }

=== TURBO (whisper.cpp) ===
${toonTurbo}

=== WHISPERX (large-v3) ===
${toonWhisperx}`;
}

function validateArbitrationResult(parsed: unknown): { transcripciones: Transcription[] } {
  const p = parsed as { transcripciones: Transcription[] };
  if (!Array.isArray(p.transcripciones)) {
    throw new Error("Estructura JSON inválida: falta 'transcripciones'");
  }
  return p;
}

/**
 * Validate that each arbitrated chunk's timestamps match either turbo or whisperx.
 * If a chunk doesn't match either, snap it to the corresponding whisperx chunk's timestamps.
 */
function validateAndSnapTimestamps(
  arbitrated: Transcription[],
  turbo: Transcription[],
  whisperx: Transcription[]
): Transcription[] {
  return arbitrated.map((chunk, idx) => {
    const turboMatch = turbo.some(
      (t) => t.inicio === chunk.inicio && t.fin === chunk.fin
    );
    const whisperxMatch = whisperx.some(
      (w) => w.inicio === chunk.inicio && w.fin === chunk.fin
    );

    if (turboMatch || whisperxMatch) {
      return chunk;
    }

    // Snap to whisperx chunk at same index if available, otherwise keep as-is
    const whisperxChunk = whisperx[idx];
    if (whisperxChunk) {
      return {
        inicio: whisperxChunk.inicio,
        fin: whisperxChunk.fin,
        texto: chunk.texto,
      };
    }

    return chunk;
  });
}

export async function transcribeEnsemble(
  audioPath: string,
  config: PipelineConfig
): Promise<EnsembleResult> {
  console.log("Ensemble: corriendo whisper.cpp turbo + WhisperX en paralelo...");

  const [turboSettled, whisperxSettled] = await Promise.allSettled([
    transcribeLocal(audioPath, config),
    transcribeWhisperX(audioPath, config, config.whisperxModel),
  ]);

  const turboOk = turboSettled.status === "fulfilled";
  const whisperxOk = whisperxSettled.status === "fulfilled";

  if (!turboOk && !whisperxOk) {
    const turboErr = turboSettled.status === "rejected" ? turboSettled.reason : null;
    const wxErr = whisperxSettled.status === "rejected" ? whisperxSettled.reason : null;
    throw new Error(
      `Ensemble: ambos transcriptores fallaron.\n` +
      `  whisper.cpp: ${turboErr instanceof Error ? turboErr.message : String(turboErr)}\n` +
      `  WhisperX: ${wxErr instanceof Error ? wxErr.message : String(wxErr)}`
    );
  }

  if (turboOk && !whisperxOk) {
    const wxErr = whisperxSettled.status === "rejected" ? whisperxSettled.reason : null;
    console.warn(
      `  Ensemble: WhisperX falló (${wxErr instanceof Error ? wxErr.message : String(wxErr)}). Usando solo whisper.cpp turbo.`
    );
    const turboResult = turboSettled.value;
    const turboChunks = whisperResultToTranscriptions(turboResult);
    return {
      whisperResult: turboResult,
      arbitrated: turboChunks,
      turbo: turboChunks,
      whisperx: [],
      arbitrationUsed: false,
    };
  }

  if (!turboOk && whisperxOk) {
    const turboErr = turboSettled.status === "rejected" ? turboSettled.reason : null;
    console.warn(
      `  Ensemble: whisper.cpp falló (${turboErr instanceof Error ? turboErr.message : String(turboErr)}). Usando solo WhisperX.`
    );
    const whisperxResult = whisperxSettled.value;
    const whisperxChunks = whisperxResultToTranscriptions(whisperxResult, 120);
    return {
      whisperResult: whisperxResult,
      arbitrated: whisperxChunks,
      turbo: [],
      whisperx: whisperxChunks,
      arbitrationUsed: false,
    };
  }

  // Both succeeded — run arbitration (narrowing guaranteed by turboOk && whisperxOk checks above)
  if (turboSettled.status !== "fulfilled" || whisperxSettled.status !== "fulfilled") {
    throw new Error("Ensemble: estado inesperado tras allSettled");
  }
  const turboResult = turboSettled.value;
  const whisperxResult = whisperxSettled.value;

  const turboChunks = whisperResultToTranscriptions(turboResult);
  const whisperxChunks = whisperxResultToTranscriptions(whisperxResult, 120);

  const toonTurbo = encode({ turbo: turboChunks });
  const toonWhisperx = encode({ whisperx: whisperxChunks });

  const arbitrationPrompt = buildArbitrationPrompt(toonTurbo, toonWhisperx);

  const arbitrationResponse = await callCodexWithSchema(
    arbitrationPrompt,
    ARBITRATION_SCHEMA,
    validateArbitrationResult,
    {
      maxRetries: 3,
      timeoutMs: 20 * 60 * 1000,
      errorCode: "E_ARBITRATION_FAILED",
    }
  );

  if (arbitrationResponse === null) {
    console.warn(
      "  Ensemble: arbitración falló. Usando WhisperX como fallback."
    );
    return {
      whisperResult: whisperxResult,
      arbitrated: whisperxChunks,
      turbo: turboChunks,
      whisperx: whisperxChunks,
      arbitrationUsed: false,
    };
  }

  const validated = validateAndSnapTimestamps(
    arbitrationResponse.transcripciones,
    turboChunks,
    whisperxChunks
  );

  return {
    whisperResult: whisperxResult,
    arbitrated: validated,
    turbo: turboChunks,
    whisperx: whisperxChunks,
    arbitrationUsed: true,
  };
}

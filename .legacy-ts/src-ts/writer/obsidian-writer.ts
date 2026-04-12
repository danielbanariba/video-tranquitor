import { writeFileSync, existsSync } from "fs";
import { join, basename, extname } from "path";
import type { PipelineResult, PipelineConfig, AttributedSegment } from "../types.js";

const DEFAULT_VAULT_PATH = "/home/banar/Desktop/obsidian/Farinter/07-Reuniones";

function formatSeconds(totalSeconds: number): string {
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = Math.floor(totalSeconds % 60);
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":");
}

function formatDateFromMs(ms: number): string {
  const d = new Date(ms);
  const year = d.getFullYear();
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function extractParticipants(transcription: AttributedSegment[]): string[] {
  const seen = new Set<string>();
  for (const seg of transcription) {
    if (seg.speaker) {
      seen.add(seg.speaker);
    }
  }
  return Array.from(seen).sort();
}

function buildFrontmatter(
  fecha: string,
  audioDurationSec: number,
  participants: string[]
): string {
  const participantsYaml =
    participants.length > 0
      ? `[${participants.map((p) => JSON.stringify(p)).join(", ")}]`
      : "[]";

  return [
    "---",
    "tags: [reunion, transcripcion]",
    `fecha: ${fecha}`,
    "estado: completado",
    `duracion: "${formatSeconds(audioDurationSec)}"`,
    `participantes: ${participantsYaml}`,
    "---",
  ].join("\n");
}

function buildTranscriptionSection(transcription: AttributedSegment[]): string {
  const lines = transcription.map((seg) => {
    const time = formatSeconds(seg.start);
    if (seg.speaker) {
      return `**${seg.speaker}** (${time}): ${seg.text}\n`;
    }
    return `(${time}): ${seg.text}\n`;
  });

  return "## Transcripción\n\n" + lines.join("\n");
}

export function writeObsidianNote(
  result: PipelineResult,
  config: PipelineConfig
): string {
  const vaultPath = config.obsidianVaultPath || DEFAULT_VAULT_PATH;

  if (!existsSync(vaultPath)) {
    throw new Error(`E_VAULT_NOT_FOUND: No se encontró la ruta del vault de Obsidian: ${vaultPath}`);
  }

  const fecha = formatDateFromMs(Date.now());
  const baseName = basename(result.inputFile, extname(result.inputFile));
  const fileName = `${fecha} ${baseName}.md`;
  const outputPath = join(vaultPath, fileName);

  const participants = extractParticipants(result.transcription);

  const sections: string[] = [];

  // Frontmatter
  sections.push(buildFrontmatter(fecha, result.audioDurationSec, participants));

  // Title
  sections.push(`\n# Reunión ${fecha} - ${baseName}\n`);

  // Transcription
  if (result.transcription.length > 0) {
    sections.push("\n" + buildTranscriptionSection(result.transcription));
  }

  // Analysis sections (only if analysis exists)
  const analysis = result.analysis;
  if (analysis) {
    // Resumen
    if (analysis.resumen) {
      sections.push("\n## Resumen\n");
      sections.push(analysis.resumen);
    }

    // Requerimientos
    if (analysis.requerimientos.length > 0) {
      sections.push("\n## Requerimientos\n");
      const reqLines = analysis.requerimientos.map(
        (r) => `- [ ] **${r.id}** (${r.prioridad}): ${r.descripcion}`
      );
      sections.push(reqLines.join("\n"));
    }

    // Accionables
    if (analysis.accionables.length > 0) {
      sections.push("\n## Accionables\n");
      const acLines = analysis.accionables.map((a) => {
        const fechaPart = a.fecha ? ` — ${a.fecha}` : "";
        return `- [ ] **${a.responsable}**: ${a.tarea}${fechaPart}`;
      });
      sections.push(acLines.join("\n"));
    }

    // Decisiones
    if (analysis.decisiones.length > 0) {
      sections.push("\n## Decisiones\n");
      const decLines = analysis.decisiones.map((d) => `- ${d}`);
      sections.push(decLines.join("\n"));
    }
  }

  const content = sections.join("\n") + "\n";
  writeFileSync(outputPath, content, "utf-8");

  return outputPath;
}

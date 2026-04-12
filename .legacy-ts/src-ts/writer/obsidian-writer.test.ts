import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdirSync, rmSync, readFileSync, existsSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { writeObsidianNote } from "./obsidian-writer.js";
import type {
  PipelineResult,
  PipelineConfig,
  AttributedSegment,
  AnalysisResult,
} from "../types.js";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeTempDir(): string {
  const dir = join(tmpdir(), `obsidian-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(dir, { recursive: true });
  return dir;
}

function makeConfig(vaultPath: string): PipelineConfig {
  return {
    watchDir: "./Audios",
    outputDir: "./output",
    transcriber: "openai",
    whisperCppPath: "",
    whisperModelPath: "",
    enableDiarization: false,
    enableAnalysis: true,
    enableObsidian: true,
    enableToon: false,
    obsidianVaultPath: vaultPath,
    hfToken: "",
    openaiApiKey: "sk-test",
    audioFilter: "",
    language: "es",
    transcriptionPrompt: "",
    transcribeModel: "gpt-4o-transcribe",
    whisperxModel: "large-v3",
    targetSampleRate: 16000,
  };
}

function makeSegment(
  speaker: string | null,
  text: string,
  start: number,
  end: number
): AttributedSegment {
  return { speaker, text, start, end };
}

function makeAnalysis(partial: Partial<AnalysisResult> = {}): AnalysisResult {
  return {
    resumen: "Resumen de la reunion.",
    requerimientos: [
      { id: "REQ-001", descripcion: "Login con OAuth", prioridad: "alta" },
    ],
    accionables: [
      { responsable: "Ana", tarea: "Configurar CI/CD", fecha: "2026-04-15" },
    ],
    decisiones: ["Migrar a PostgreSQL", "Adoptar TypeScript estricto"],
    ...partial,
  };
}

function makeResult(
  override: Partial<PipelineResult> = {},
  inputFile = "/videos/reunion_semanal.mp4"
): PipelineResult {
  return {
    inputFile,
    wavPath: "/tmp/reunion_semanal.wav",
    transcription: [
      makeSegment("SPEAKER_00", "Buenos dias a todos", 10, 13),
      makeSegment("SPEAKER_01", "Hola como estan", 14, 17),
    ],
    analysis: makeAnalysis(),
    toonOutputPath: null,
    obsidianOutputPath: null,
    durationMs: 1200000,
    audioDurationSec: 1200,
    stagesRun: ["preprocess", "transcribe", "analyze"],
    whisperResult: null,
    ...override,
  };
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe("writeObsidianNote", () => {
  let vaultDir: string;

  beforeEach(() => {
    vaultDir = makeTempDir();
  });

  afterEach(() => {
    if (existsSync(vaultDir)) {
      rmSync(vaultDir, { recursive: true, force: true });
    }
  });

  // 1. Full AnalysisResult → snapshot + verify YAML frontmatter fields
  it("generates markdown with all 5 required YAML frontmatter fields", () => {
    const result = makeResult();
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    // Snapshot test
    expect(content).toMatchSnapshot();

    // Frontmatter must contain exactly these 5 fields
    expect(content).toMatch(/^tags:/m);
    expect(content).toMatch(/^fecha:/m);
    expect(content).toMatch(/^estado:/m);
    expect(content).toMatch(/^duracion:/m);
    expect(content).toMatch(/^participantes:/m);
  });

  it("includes all analysis sections in the full output", () => {
    const result = makeResult();
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("## Resumen");
    expect(content).toContain("## Requerimientos");
    expect(content).toContain("## Accionables");
    expect(content).toContain("## Decisiones");
  });

  // 2. No analysis (null) → skips Resumen/Requerimientos/Accionables/Decisiones
  it("omits analysis sections when analysis is null", () => {
    const result = makeResult({ analysis: null });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("## Transcripción");
    expect(content).not.toContain("## Resumen");
    expect(content).not.toContain("## Requerimientos");
    expect(content).not.toContain("## Accionables");
    expect(content).not.toContain("## Decisiones");
  });

  // 3. Analysis with empty arrays → sections omitted cleanly
  it("omits empty array sections (requerimientos, accionables, decisiones)", () => {
    const result = makeResult({
      analysis: makeAnalysis({
        requerimientos: [],
        accionables: [],
        decisiones: [],
      }),
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("## Resumen");
    expect(content).not.toContain("## Requerimientos");
    expect(content).not.toContain("## Accionables");
    expect(content).not.toContain("## Decisiones");
  });

  // 4. Transcription with speakers → `**SPEAKER_00** (00:00:10): text`
  it("renders transcription segments with speaker attribution", () => {
    const result = makeResult({
      transcription: [makeSegment("SPEAKER_00", "Buenos dias", 10, 13)],
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("**SPEAKER_00** (00:00:10): Buenos dias");
  });

  // 5. Transcription without speakers → `(00:00:10): text`
  it("renders transcription segments without speaker when speaker is null", () => {
    const result = makeResult({
      transcription: [makeSegment(null, "Texto sin hablante", 10, 13)],
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("(00:00:10): Texto sin hablante");
    expect(content).not.toMatch(/\*\*null\*\*/);
  });

  it("formats timestamps correctly in transcription", () => {
    const result = makeResult({
      transcription: [
        makeSegment("SPEAKER_00", "Primero", 0, 2),
        makeSegment("SPEAKER_01", "Segundo", 3661, 3665), // 1h 1m 1s
      ],
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("**SPEAKER_00** (00:00:00): Primero");
    expect(content).toContain("**SPEAKER_01** (01:01:01): Segundo");
  });

  // 6. Vault path missing → throws E_VAULT_NOT_FOUND
  it("throws E_VAULT_NOT_FOUND when vault path does not exist", () => {
    const result = makeResult();
    const config = makeConfig("/absolutely/nonexistent/vault/path/12345");

    expect(() => writeObsidianNote(result, config)).toThrow("E_VAULT_NOT_FOUND");
  });

  // Extra: participants are extracted from transcription and populated in frontmatter
  it("populates participantes in YAML frontmatter from transcription speakers", () => {
    const result = makeResult({
      transcription: [
        makeSegment("SPEAKER_00", "Hola", 0, 2),
        makeSegment("SPEAKER_01", "Mundo", 3, 5),
      ],
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("SPEAKER_00");
    expect(content).toContain("SPEAKER_01");
  });

  // Extra: output file is created inside the vault dir
  it("writes the file inside the vault directory", () => {
    const result = makeResult();
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);

    expect(outputPath.startsWith(vaultDir)).toBe(true);
    expect(existsSync(outputPath)).toBe(true);
  });

  // Extra: file name includes date and input file base name
  it("names the output file with date and input base name", () => {
    const result = makeResult({}, "/videos/mi_reunion.mp4");
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const fileName = outputPath.split("/").pop()!;

    expect(fileName).toMatch(/^\d{4}-\d{2}-\d{2} mi_reunion\.md$/);
  });

  // Extra: requerimientos rendered with checkbox format
  it("renders requerimientos with checkbox and priority", () => {
    const result = makeResult({
      analysis: makeAnalysis({
        requerimientos: [{ id: "REQ-001", descripcion: "Auth flow", prioridad: "alta" }],
      }),
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("- [ ] **REQ-001** (alta): Auth flow");
  });

  // Extra: accionables rendered with responsable and optional fecha
  it("renders accionables with responsable, tarea and fecha when present", () => {
    const result = makeResult({
      analysis: makeAnalysis({
        accionables: [
          { responsable: "Juan", tarea: "Revisar PR", fecha: "2026-04-20" },
        ],
      }),
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("- [ ] **Juan**: Revisar PR — 2026-04-20");
  });

  it("renders accionables without fecha when fecha is absent", () => {
    const result = makeResult({
      analysis: makeAnalysis({
        accionables: [{ responsable: "Maria", tarea: "Documentar API" }],
      }),
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("- [ ] **Maria**: Documentar API");
    // No trailing " — " when fecha is absent
    expect(content).not.toContain("Maria**: Documentar API —");
  });

  // Extra: decisiones rendered as bullet list
  it("renders decisiones as bullet list", () => {
    const result = makeResult({
      analysis: makeAnalysis({
        decisiones: ["Usar Redis para caché"],
      }),
    });
    const config = makeConfig(vaultDir);

    const outputPath = writeObsidianNote(result, config);
    const content = readFileSync(outputPath, "utf-8");

    expect(content).toContain("- Usar Redis para caché");
  });
});

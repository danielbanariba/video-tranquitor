import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

// We need to intercept dotenv's config() call so it doesn't read from .env files
// and pollute process.env during tests.
vi.mock("dotenv", () => ({
  config: vi.fn(), // no-op
}));

// Import AFTER mocking dotenv
const { loadConfig } = await import("./config.js");

// ---------------------------------------------------------------------------
// Snapshot of original env — restored after every test
// ---------------------------------------------------------------------------

let originalEnv: NodeJS.ProcessEnv;

beforeEach(() => {
  originalEnv = { ...process.env };
  // Clear all relevant vars so tests start from a clean slate
  const varsToDelete = [
    "OPENAI_API_KEY",
    "TRANSCRIBER",
    "WHISPER_CPP_PATH",
    "WHISPER_MODEL_PATH",
    "ENABLE_DIARIZATION",
    "HF_TOKEN",
    "ENABLE_ANALYSIS",
    "ENABLE_OBSIDIAN",
    "ENABLE_TOON",
    "WATCH_DIR",
    "OUTPUT_DIR",
    "OBSIDIAN_VAULT_PATH",
    "AUDIO_FILTER",
    "LANGUAGE",
    "TRANSCRIPTION_PROMPT",
    "OPENAI_TRANSCRIBE_MODEL",
    "TARGET_SAMPLE_RATE",
  ];
  for (const v of varsToDelete) {
    delete process.env[v];
  }
});

afterEach(() => {
  process.env = originalEnv;
});

// ---------------------------------------------------------------------------
// Minimal valid env (only OPENAI_API_KEY + transcriber=openai to skip local checks)
// ---------------------------------------------------------------------------

function setMinimalValidEnv() {
  process.env.OPENAI_API_KEY = "sk-test-key";
  process.env.TRANSCRIBER = "openai"; // skip local whisper requirements
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("loadConfig", () => {
  // 1. All defaults
  it("returns a valid config with defaults when only OPENAI_API_KEY is set (openai transcriber)", () => {
    setMinimalValidEnv();
    const cfg = loadConfig();

    expect(cfg.transcriber).toBe("openai");
    expect(cfg.watchDir).toBe("./Audios");
    expect(cfg.outputDir).toBe("./output");
    expect(cfg.enableDiarization).toBe(false);
    expect(cfg.enableAnalysis).toBe(true);
    expect(cfg.enableObsidian).toBe(true);
    expect(cfg.enableToon).toBe(true);
    expect(cfg.language).toBe("es");
    expect(cfg.targetSampleRate).toBe(16000);
    expect(cfg.transcribeModel).toBe("gpt-4o-transcribe");
    expect(cfg.openaiApiKey).toBe("sk-test-key");
  });

  // 2. OPENAI_API_KEY missing → throws
  it("throws when OPENAI_API_KEY is missing", () => {
    delete process.env.OPENAI_API_KEY;
    expect(() => loadConfig()).toThrow("OPENAI_API_KEY");
  });

  // 3. TRANSCRIBER=local + missing WHISPER_CPP_PATH → throws
  it("throws when TRANSCRIBER=local and WHISPER_CPP_PATH is missing", () => {
    process.env.OPENAI_API_KEY = "sk-test-key";
    process.env.TRANSCRIBER = "local";
    // WHISPER_CPP_PATH not set
    expect(() => loadConfig()).toThrow("WHISPER_CPP_PATH");
  });

  // 4. TRANSCRIBER=local + missing WHISPER_MODEL_PATH → throws
  it("throws when TRANSCRIBER=local and WHISPER_MODEL_PATH is missing", () => {
    process.env.OPENAI_API_KEY = "sk-test-key";
    process.env.TRANSCRIBER = "local";
    process.env.WHISPER_CPP_PATH = "/usr/local/bin/whisper";
    // WHISPER_MODEL_PATH not set
    expect(() => loadConfig()).toThrow("WHISPER_MODEL_PATH");
  });

  // 5. ENABLE_DIARIZATION=true + missing HF_TOKEN → throws
  it("throws when ENABLE_DIARIZATION=true and HF_TOKEN is missing", () => {
    process.env.OPENAI_API_KEY = "sk-test-key";
    process.env.TRANSCRIBER = "openai";
    process.env.ENABLE_DIARIZATION = "true";
    // HF_TOKEN not set
    expect(() => loadConfig()).toThrow("HF_TOKEN");
  });

  // 6. All env vars set → returns config with all values populated
  it("returns fully populated config when all env vars are set", () => {
    process.env.OPENAI_API_KEY = "sk-real-key";
    process.env.TRANSCRIBER = "local";
    process.env.WHISPER_CPP_PATH = "/opt/whisper/main";
    process.env.WHISPER_MODEL_PATH = "/opt/whisper/models/ggml-base.bin";
    process.env.ENABLE_DIARIZATION = "true";
    process.env.HF_TOKEN = "hf_abc123";
    process.env.WATCH_DIR = "/videos";
    process.env.OUTPUT_DIR = "/out";
    process.env.OBSIDIAN_VAULT_PATH = "/vault/notes";
    process.env.LANGUAGE = "en";
    process.env.OPENAI_TRANSCRIBE_MODEL = "gpt-4o-audio-preview";
    process.env.TARGET_SAMPLE_RATE = "44100";
    process.env.ENABLE_ANALYSIS = "true";
    process.env.ENABLE_OBSIDIAN = "true";
    process.env.ENABLE_TOON = "true";

    const cfg = loadConfig();

    expect(cfg.openaiApiKey).toBe("sk-real-key");
    expect(cfg.transcriber).toBe("local");
    expect(cfg.whisperCppPath).toBe("/opt/whisper/main");
    expect(cfg.whisperModelPath).toBe("/opt/whisper/models/ggml-base.bin");
    expect(cfg.enableDiarization).toBe(true);
    expect(cfg.hfToken).toBe("hf_abc123");
    expect(cfg.watchDir).toBe("/videos");
    expect(cfg.outputDir).toBe("/out");
    expect(cfg.obsidianVaultPath).toBe("/vault/notes");
    expect(cfg.language).toBe("en");
    expect(cfg.transcribeModel).toBe("gpt-4o-audio-preview");
    expect(cfg.targetSampleRate).toBe(44100);
  });

  // 7a. ENABLE_ANALYSIS="false" → enableAnalysis: false
  it("sets enableAnalysis=false when ENABLE_ANALYSIS=false", () => {
    setMinimalValidEnv();
    process.env.ENABLE_ANALYSIS = "false";
    const cfg = loadConfig();
    expect(cfg.enableAnalysis).toBe(false);
  });

  // 7b. ENABLE_ANALYSIS not set (or any non-"false" value) → enableAnalysis: true
  it("sets enableAnalysis=true when ENABLE_ANALYSIS is not set", () => {
    setMinimalValidEnv();
    delete process.env.ENABLE_ANALYSIS;
    const cfg = loadConfig();
    expect(cfg.enableAnalysis).toBe(true);
  });

  it("sets enableAnalysis=true when ENABLE_ANALYSIS=true", () => {
    setMinimalValidEnv();
    process.env.ENABLE_ANALYSIS = "true";
    const cfg = loadConfig();
    expect(cfg.enableAnalysis).toBe(true);
  });

  // 7c. Boolean parsing for ENABLE_OBSIDIAN
  it("sets enableObsidian=false when ENABLE_OBSIDIAN=false", () => {
    setMinimalValidEnv();
    process.env.ENABLE_OBSIDIAN = "false";
    const cfg = loadConfig();
    expect(cfg.enableObsidian).toBe(false);
  });

  // 7d. Boolean parsing for ENABLE_TOON
  it("sets enableToon=false when ENABLE_TOON=false", () => {
    setMinimalValidEnv();
    process.env.ENABLE_TOON = "false";
    const cfg = loadConfig();
    expect(cfg.enableToon).toBe(false);
  });

  // ENABLE_DIARIZATION true/false parsing
  it("sets enableDiarization=true when ENABLE_DIARIZATION=true and HF_TOKEN is present", () => {
    setMinimalValidEnv();
    process.env.ENABLE_DIARIZATION = "true";
    process.env.HF_TOKEN = "hf_xyz";
    const cfg = loadConfig();
    expect(cfg.enableDiarization).toBe(true);
    expect(cfg.hfToken).toBe("hf_xyz");
  });

  it("sets enableDiarization=false when ENABLE_DIARIZATION is not set", () => {
    setMinimalValidEnv();
    const cfg = loadConfig();
    expect(cfg.enableDiarization).toBe(false);
  });
});

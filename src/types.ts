export interface PipelineConfig {
  watchDir: string;
  outputDir: string;
  transcriber: "local" | "openai" | "whisperx" | "ensemble";
  whisperxModel: string;
  whisperCppPath: string;
  whisperModelPath: string;
  enableDiarization: boolean;
  enableAnalysis: boolean;
  enableObsidian: boolean;
  enableToon: boolean;
  obsidianVaultPath: string;
  hfToken: string;
  openaiApiKey: string;
  audioFilter: string;
  language: string;
  transcriptionPrompt: string;
  transcribeModel: string;
  targetSampleRate: number;
}

export interface WhisperWord {
  word: string;
  start: number;
  end: number;
}

export interface WhisperSegment {
  text: string;
  start: number;
  end: number;
  words: WhisperWord[];
}

export interface WhisperResult {
  segments: WhisperSegment[];
  language: string;
}

export interface DiarizationSegment {
  speaker: string;
  start: number;
  end: number;
}

export interface AttributedSegment {
  speaker: string | null;
  text: string;
  start: number;
  end: number;
}

export interface AnalysisResult {
  resumen: string;
  requerimientos: { id: string; descripcion: string; prioridad: string; }[];
  accionables: { responsable: string; tarea: string; fecha?: string; }[];
  decisiones: string[];
}

export interface Transcription {
  inicio: string;
  fin: string;
  texto: string;
}

export interface PipelineResult {
  inputFile: string;
  wavPath: string;
  transcription: AttributedSegment[];
  analysis: AnalysisResult | null;
  toonOutputPath: string | null;
  obsidianOutputPath: string | null;
  durationMs: number;
  audioDurationSec: number;
  stagesRun: string[];
  whisperResult: WhisperResult | null;
}

export interface EnsembleResult {
  whisperResult: WhisperResult;
  arbitrated: Transcription[];
  turbo: Transcription[];
  whisperx: Transcription[];
  arbitrationUsed: boolean;
}

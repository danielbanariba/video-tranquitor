import { writeFileSync } from "fs";
import { encode } from "@toon-format/toon";
import type { Transcription } from "../types.js";

export function writeToon(data: Transcription[], outputPath: string): void {
  const toonContent = encode({ transcripciones: data });
  writeFileSync(outputPath, toonContent, "utf-8");
}

import { execSync } from "child_process";
import { statSync } from "fs";

export function preprocessAudio(
  inputPath: string,
  outputPath: string,
  audioFilter: string,
  targetSampleRate: number
): boolean {
  const baseCommand = `ffmpeg -i "${inputPath}" -vn -ac 1 -ar ${targetSampleRate} -sample_fmt s16`;
  const filterArg = audioFilter ? ` -af "${audioFilter}"` : "";
  try {
    execSync(`${baseCommand}${filterArg} -y "${outputPath}"`, { stdio: "pipe" });
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

export function getAudioDuration(audioPath: string): number {
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

export function getFileSizeMB(filePath: string): number {
  return statSync(filePath).size / (1024 * 1024);
}

export function formatTime(seconds: number): string {
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);
  return `${hours.toString().padStart(2, "0")}:${minutes.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
}

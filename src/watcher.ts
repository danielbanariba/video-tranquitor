import { mkdirSync, existsSync } from "fs";
import { extname, basename } from "path";
import chokidar from "chokidar";
import type { PipelineConfig } from "./types.js";
import { VIDEO_EXTENSIONS, AUDIO_EXTENSIONS } from "./pipeline.js";

const DEBOUNCE_MS = 500;

export function startWatcher(
  config: PipelineConfig,
  onFile: (filePath: string) => Promise<void>
): void {
  if (!existsSync(config.watchDir)) {
    mkdirSync(config.watchDir, { recursive: true });
    console.log(`Directorio creado: ${config.watchDir}`);
  }

  const pending = new Map<string, ReturnType<typeof setTimeout>>();

  const watcher = chokidar.watch(config.watchDir, {
    persistent: true,
    ignoreInitial: true,
    awaitWriteFinish: {
      stabilityThreshold: 1000,
      pollInterval: 200,
    },
  });

  watcher.on("add", (filePath: string) => {
    const name = basename(filePath);
    const ext = extname(filePath).toLowerCase();

    // Ignorar archivos ocultos y temporales
    if (name.startsWith(".") || name.startsWith("temp_")) {
      return;
    }

    // Solo procesar extensiones soportadas
    if (!VIDEO_EXTENSIONS.has(ext) && !AUDIO_EXTENSIONS.has(ext)) {
      return;
    }

    // Debounce para archivos que se están copiando
    const existing = pending.get(filePath);
    if (existing) {
      clearTimeout(existing);
    }

    const timer = setTimeout(async () => {
      pending.delete(filePath);
      console.log(`\nArchivo detectado: ${filePath}`);
      try {
        await onFile(filePath);
      } catch (error) {
        console.error(`Error al procesar ${filePath}:`, error);
      }
    }, DEBOUNCE_MS);

    pending.set(filePath, timer);
  });

  watcher.on("error", (error: unknown) => {
    console.error("Error en el watcher:", error);
  });

  console.log(`Watcher iniciado. Monitoreando: ${config.watchDir}`);
  console.log("Esperando archivos de audio/video... (Ctrl+C para detener)");

  const shutdown = (): void => {
    console.log("\nDeteniendo watcher...");
    void watcher.close().then(() => {
      console.log("Watcher detenido.");
      process.exit(0);
    });
  };

  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);
}

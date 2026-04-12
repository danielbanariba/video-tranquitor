import { existsSync } from "fs";
import { loadConfig } from "./config.js";
import { runPipeline } from "./pipeline.js";
import { startWatcher } from "./watcher.js";

function parseArgs(): { mode: "file" | "watch"; filePath?: string } {
  const args = process.argv.slice(2);
  const fileIndex = args.indexOf("--file");

  if (fileIndex !== -1) {
    const filePath = args[fileIndex + 1];
    if (!filePath || filePath.startsWith("--")) {
      console.error("Error: --file requiere una ruta de archivo válida.");
      console.error("Uso: npm run process -- --file Audios/archivo.mp4");
      process.exit(1);
    }
    return { mode: "file", filePath };
  }

  if (args.includes("--watch")) {
    return { mode: "watch" };
  }

  if (args.length > 0 && !args[0].startsWith("--")) {
    return { mode: "file", filePath: args[0] };
  }

  return { mode: "watch" };
}

async function main(): Promise<void> {
  const { mode, filePath } = parseArgs();

  let config;
  try {
    config = loadConfig();
  } catch (error) {
    console.error("Error de configuración:", error instanceof Error ? error.message : error);
    process.exit(1);
  }

  if (mode === "file" && filePath) {
    if (!existsSync(filePath)) {
      console.error(`Error: El archivo no existe: ${filePath}`);
      process.exit(1);
    }
    try {
      const result = await runPipeline(filePath, config);
      console.log(`\nProcesamiento completado. Etapas ejecutadas: ${result.stagesRun.join(", ")}`);
    } catch (error) {
      console.error("Error durante el procesamiento:", error instanceof Error ? error.message : error);
      process.exit(1);
    }
  } else {
    startWatcher(config, async (detectedFile) => {
      try {
        await runPipeline(detectedFile, config);
      } catch (error) {
        console.error(
          `Error al procesar ${detectedFile}:`,
          error instanceof Error ? error.message : error
        );
      }
    });
  }
}

await main();

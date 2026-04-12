import { spawn } from "child_process";
import { writeFileSync, readFileSync, existsSync, unlinkSync, mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const DEFAULT_MAX_RETRIES = 3;
const DEFAULT_TIMEOUT_MS = 20 * 60 * 1000; // 20 minutes
const DEFAULT_ERROR_CODE = "E_CODEX_FAILED";
const INITIAL_BACKOFF_MS = 2000;

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractJson(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    return trimmed;
  }

  const fenceMatch = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/);
  if (fenceMatch) {
    return fenceMatch[1].trim();
  }

  const firstBrace = trimmed.indexOf("{");
  const lastBrace = trimmed.lastIndexOf("}");
  if (firstBrace !== -1 && lastBrace > firstBrace) {
    return trimmed.slice(firstBrace, lastBrace + 1);
  }

  return trimmed;
}

async function callCodex(prompt: string, schema: object, timeoutMs: number): Promise<string> {
  const tmpDir = mkdtempSync(join(tmpdir(), "codex-client-"));
  const schemaPath = join(tmpDir, "schema.json");
  const outputPath = join(tmpDir, "output.txt");

  writeFileSync(schemaPath, JSON.stringify(schema), "utf-8");

  try {
    await new Promise<void>((resolve, reject) => {
      const child = spawn(
        "codex",
        [
          "exec",
          "--sandbox", "read-only",
          "--skip-git-repo-check",
          "--ephemeral",
          "--output-schema", schemaPath,
          "--output-last-message", outputPath,
          "--color", "never",
          "-",
        ],
        { stdio: ["pipe", "pipe", "pipe"] }
      );

      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        reject(new Error(`Codex timeout (${timeoutMs}ms)`));
      }, timeoutMs);

      let stderr = "";
      child.stderr.on("data", (chunk) => {
        stderr += chunk.toString();
      });
      child.stdout.on("data", () => {
        // drain stdout to prevent buffer blocking
      });

      child.on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      });

      child.on("close", (code) => {
        clearTimeout(timer);
        if (code === 0) {
          resolve();
        } else {
          reject(new Error(`Codex exit ${code}: ${stderr.slice(-500)}`));
        }
      });

      child.stdin.write(prompt);
      child.stdin.end();
    });

    if (!existsSync(outputPath)) {
      throw new Error("Codex no generó archivo de salida");
    }

    const content = readFileSync(outputPath, "utf-8").trim();
    if (!content) {
      throw new Error("Respuesta vacía de Codex");
    }

    return content;
  } finally {
    try {
      if (existsSync(schemaPath)) unlinkSync(schemaPath);
      if (existsSync(outputPath)) unlinkSync(outputPath);
    } catch {
      // ignore cleanup errors
    }
  }
}

export async function callCodexWithSchema<T>(
  prompt: string,
  schema: object,
  validate: (parsed: unknown) => T,
  opts?: { maxRetries?: number; timeoutMs?: number; errorCode?: string }
): Promise<T | null> {
  const maxRetries = opts?.maxRetries ?? DEFAULT_MAX_RETRIES;
  const timeoutMs = opts?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const errorCode = opts?.errorCode ?? DEFAULT_ERROR_CODE;

  let lastError: unknown = null;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const rawOutput = await callCodex(prompt, schema, timeoutMs);
      const jsonText = extractJson(rawOutput);
      const parsed = JSON.parse(jsonText) as unknown;
      return validate(parsed);
    } catch (error) {
      lastError = error;

      if (attempt < maxRetries) {
        const backoffMs = INITIAL_BACKOFF_MS * Math.pow(2, attempt - 1);
        console.warn(
          `  Codex: intento ${attempt}/${maxRetries} fallido. Reintentando en ${backoffMs / 1000}s...`
        );
        await sleep(backoffMs);
      }
    }
  }

  console.error(
    `${errorCode}: Codex fallido tras ${maxRetries} intentos.`,
    lastError instanceof Error ? lastError.message : lastError
  );

  return null;
}

import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export const ROOT_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const RESULTS_DIR = path.join(ROOT_DIR, 'results');
export const SCRIPTS_DIR = path.join(ROOT_DIR, 'scripts');

export function projectPath(...parts) {
  return path.join(ROOT_DIR, ...parts);
}

export function isMain(metaUrl) {
  return Boolean(process.argv[1])
    && path.resolve(process.argv[1]) === fileURLToPath(metaUrl);
}

export function round(value, digits = 2) {
  return Number(Number(value).toFixed(digits));
}

export function parseCsv(value) {
  return value?.split(',').map((item) => item.trim()).filter(Boolean) ?? null;
}

export function positiveInteger(value, name) {
  if (value == null) {
    return null;
  }
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return parsed;
}

export function formatDuration(ms) {
  const seconds = Math.floor(ms / 1000);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const remainder = seconds % 60;
  if (hours) return `${hours}h ${minutes}m ${remainder}s`;
  if (minutes) return `${minutes}m ${remainder}s`;
  return `${remainder}s`;
}

export async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, 'utf8'));
}

export async function writeJson(filePath, value) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
}

export async function loadBenchmarkCorpus(maxTexts = null) {
  const corpus = await readJson(projectPath('data', 'benchmark-corpus.json'));
  const documents = corpus.documents.slice(0, maxTexts ?? corpus.documents.length);
  const documentIds = new Set(documents.map((document) => document.id));
  const queryPairs = corpus.query_pairs.filter(
    (pair) => documentIds.has(pair.sv_doc_id) && documentIds.has(pair.tr_doc_id),
  );
  return { corpus, documents, queryPairs };
}

export function snapshotMemory() {
  const memory = process.memoryUsage();
  return {
    rss_mb: round(memory.rss / 1024 / 1024),
    heap_used_mb: round(memory.heapUsed / 1024 / 1024),
    external_mb: round(memory.external / 1024 / 1024),
  };
}

export class MemoryMonitor {
  constructor(intervalMs = 200) {
    this.intervalMs = intervalMs;
    this.interval = null;
    this.peak = { rss_mb: 0, heap_used_mb: 0, external_mb: 0 };
  }

  start() {
    this.sample();
    this.interval = setInterval(() => this.sample(), this.intervalMs);
    this.interval.unref();
  }

  sample() {
    const current = snapshotMemory();
    for (const key of Object.keys(this.peak)) {
      this.peak[key] = Math.max(this.peak[key], current[key]);
    }
  }

  stop() {
    clearInterval(this.interval);
    this.interval = null;
    this.sample();
    return {
      peak_rss_mb: this.peak.rss_mb,
      peak_heap_used_mb: this.peak.heap_used_mb,
      peak_external_mb: this.peak.external_mb,
    };
  }
}

export function summarizeTimings(values) {
  if (!values.length) {
    return { count: 0, mean_ms: 0, total_ms: 0, p50_ms: 0, p95_ms: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const total = values.reduce((sum, value) => sum + value, 0);
  const percentile = (ratio) =>
    sorted[Math.min(sorted.length - 1, Math.floor(ratio * (sorted.length - 1)))];
  return {
    count: values.length,
    mean_ms: round(total / values.length),
    total_ms: round(total),
    p50_ms: round(percentile(0.5)),
    p95_ms: round(percentile(0.95)),
  };
}

export function classifyRuntimeError(message) {
  const normalized = String(message).toLowerCase();
  if (
    normalized.includes('bad_alloc')
    || normalized.includes('out of memory')
    || normalized.includes('oom')
    || normalized.includes('killed')
  ) {
    return 'oom';
  }
  if (
    normalized.includes('gatherblockquantized')
    || normalized.includes('bits==4 or 8')
  ) {
    return 'gather_quant';
  }
  if (normalized.includes('shader-f16') || normalized.includes('requires f16')) {
    return 'shader_f16';
  }
  if (normalized.includes('webgpu validation')) {
    return 'webgpu_validation';
  }
  if (
    normalized.includes('external data')
    || normalized.includes('mountedfiles')
    || normalized.includes('onnx_data')
  ) {
    return 'external_data';
  }
  return 'error';
}

export async function dispose(resource) {
  if (!resource?.dispose) {
    return;
  }
  try {
    await resource.dispose();
  } catch {
    // A failed or OOM session may also fail during cleanup.
  }
}

function parseJsonOutput(output) {
  const trimmed = output.trim();
  if (!trimmed) {
    throw new Error('worker produced no JSON output');
  }
  try {
    return JSON.parse(trimmed);
  } catch {
    const lines = trimmed.split('\n');
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      try {
        return JSON.parse(lines[index]);
      } catch {
        // Keep scanning in case a dependency logged before the JSON payload.
      }
    }
    throw new Error('worker output was not valid JSON');
  }
}

export async function runJsonWorker(
  script,
  args = [],
  {
    exposeGc = true,
    resultFile = false,
    failureStatus = 'error',
    cwd = ROOT_DIR,
    env = process.env,
  } = {},
) {
  let resultPath = null;
  const workerArgs = [...args];

  if (resultFile) {
    const tmpDir = path.join(RESULTS_DIR, '.tmp');
    await fs.mkdir(tmpDir, { recursive: true });
    resultPath = path.join(
      tmpDir,
      `${path.basename(script, path.extname(script))}-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}.json`,
    );
    workerArgs.push('--result-file', resultPath);
  }

  const nodeArgs = [...(exposeGc ? ['--expose-gc'] : []), script, ...workerArgs];
  const child = spawn(process.execPath, nodeArgs, {
    cwd,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let stdout = '';
  let stderr = '';
  child.stdout.on('data', (chunk) => {
    stdout += chunk.toString();
  });
  child.stderr.on('data', (chunk) => {
    stderr += chunk.toString();
  });

  const outcome = await new Promise((resolve) => {
    child.once('error', (error) => resolve({ code: null, signal: null, error }));
    child.once('close', (code, signal) => resolve({ code, signal, error: null }));
  });

  let result;
  try {
    result = resultPath
      ? await readJson(resultPath)
      : parseJsonOutput(stdout);
  } catch (error) {
    const killed = outcome.signal === 'SIGKILL' || outcome.code === 137;
    result = {
      status: failureStatus,
      error: killed
        ? 'Process killed (likely OOM)'
        : outcome.error?.message ?? error.message,
      stderr_tail: stderr.slice(-500),
    };
  } finally {
    if (resultPath) {
      await fs.unlink(resultPath).catch(() => {});
    }
  }

  if (
    result.status !== 'ok'
    && (outcome.signal === 'SIGKILL' || outcome.code === 137)
  ) {
    result.status = failureStatus;
    result.error ??= 'Process killed (likely OOM)';
    result.error_kind ??= 'oom';
  }

  return {
    result,
    stdout,
    stderr,
    code: outcome.code,
    signal: outcome.signal,
  };
}

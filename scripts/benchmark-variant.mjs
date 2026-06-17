#!/usr/bin/env node
/**
 * Runs a single model/variant in an isolated process (survives OOM kills).
 * Writes JSON result to --result-file and exits.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';
import { computeQuality, mean } from '../lib/metrics.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--model-id') args.modelId = argv[++i];
    else if (key === '--model-name') args.modelName = argv[++i];
    else if (key === '--dtype') args.dtype = argv[++i];
    else if (key === '--variant-label') args.variantLabel = argv[++i];
    else if (key === '--model-file-name') args.modelFileName = argv[++i];
    else if (key === '--backend') args.backend = argv[++i];
    else if (key === '--max-texts') args.maxTexts = Number(argv[++i]);
    else if (key === '--result-file') args.resultFile = argv[++i];
  }
  return args;
}

function round(n, digits = 2) {
  return Number(n.toFixed(digits));
}

function snapshotMemory() {
  const m = process.memoryUsage();
  return {
    rss_mb: round(m.rss / 1024 / 1024),
    heap_used_mb: round(m.heapUsed / 1024 / 1024),
    external_mb: round(m.external / 1024 / 1024),
  };
}

class MemoryMonitor {
  constructor() {
    this.peak = { rss_mb: 0, heap_used_mb: 0, external_mb: 0 };
    this.interval = null;
  }
  start() {
    this.sample();
    this.interval = setInterval(() => this.sample(), 200);
  }
  sample() {
    const s = snapshotMemory();
    for (const key of Object.keys(this.peak)) {
      if (s[key] > this.peak[key]) this.peak[key] = s[key];
    }
  }
  stop() {
    if (this.interval) clearInterval(this.interval);
    this.sample();
    return { peak_rss_mb: this.peak.rss_mb, peak_heap_used_mb: this.peak.heap_used_mb, peak_external_mb: this.peak.external_mb };
  }
}

function summarizeTimings(values) {
  const sum = values.reduce((a, b) => a + b, 0);
  return {
    count: values.length,
    mean_ms: round(sum / values.length),
    total_ms: round(sum),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const corpus = JSON.parse(
    await fs.readFile(path.join(root, 'data', 'benchmark-corpus.json'), 'utf8'),
  );

  let documents = corpus.documents;
  if (args.maxTexts) {
    documents = documents.slice(0, args.maxTexts);
  }

  const docIds = new Set(documents.map((d) => d.id));
  const queryPairs = corpus.query_pairs.filter(
    (p) => docIds.has(p.sv_doc_id) && docIds.has(p.tr_doc_id),
  );

  const wallStart = performance.now();
  const mem = new MemoryMonitor();
  mem.start();

  const result = {
    model_id: args.modelId,
    model_name: args.modelName,
    variant: args.variantLabel,
    dtype: args.dtype,
    model_file_name: args.modelFileName ?? null,
    backend_requested: args.backend ?? 'cpu',
    backend_used: null,
    status: 'pending',
    started_at: new Date().toISOString(),
    memory_at_start: snapshotMemory(),
    load_time_ms: null,
    total_time_ms: null,
    embedding_dim: null,
    inference: null,
    quality: null,
    memory: null,
    error: null,
  };

  let extractor;
  const loadStart = performance.now();

  try {
    const options = { dtype: args.dtype };
    if (args.modelFileName) {
      options.model_file_name = args.modelFileName;
    }

    extractor = await createFeatureExtractor(args.modelId, options, args.backend ?? 'cpu');
    result.backend_used = extractor._benchmark_backend ?? args.backend ?? 'auto';
    result.load_time_ms = round(performance.now() - loadStart);

    const latencies = [];
    const embeddings = new Map();

    for (const doc of documents) {
      const t0 = performance.now();
      const tensor = await extractor(doc.text, { pooling: 'mean', normalize: true });
      const vector = tensor.tolist()[0];
      latencies.push(performance.now() - t0);
      embeddings.set(doc.id, vector);
      result.embedding_dim ??= vector.length;
    }

    result.inference = summarizeTimings(latencies);
    result.quality = computeQuality(embeddings, documents, queryPairs);
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = error instanceof Error ? error.message : String(error);
    result.load_time_ms = round(performance.now() - loadStart);
  } finally {
    if (extractor) await extractor.dispose();
    if (global.gc) global.gc();
    result.memory = mem.stop();
    result.total_time_ms = round(performance.now() - wallStart);
    result.finished_at = new Date().toISOString();
  }

  await fs.writeFile(args.resultFile, JSON.stringify(result, null, 2));
}

main().catch(async (error) => {
  const args = parseArgs(process.argv);
  const payload = {
    status: 'error',
    error: error instanceof Error ? error.message : String(error),
    finished_at: new Date().toISOString(),
  };
  if (args.resultFile) {
    await fs.writeFile(args.resultFile, JSON.stringify(payload, null, 2));
  }
  process.exit(1);
});

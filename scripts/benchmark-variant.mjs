#!/usr/bin/env node
import { parseArgs } from 'node:util';
import {
  MemoryMonitor,
  dispose,
  loadBenchmarkCorpus,
  positiveInteger,
  round,
  snapshotMemory,
  summarizeTimings,
  writeJson,
} from '../lib/benchmark-support.mjs';
import { computeQuality } from '../lib/metrics.mjs';
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';

function parseCli() {
  const { values } = parseArgs({
    options: {
      'model-id': { type: 'string' },
      'model-name': { type: 'string' },
      dtype: { type: 'string' },
      'variant-label': { type: 'string' },
      'model-file-name': { type: 'string' },
      backend: { type: 'string' },
      'max-texts': { type: 'string' },
      'result-file': { type: 'string' },
    },
    strict: true,
  });
  for (const required of ['model-id', 'model-name', 'dtype', 'variant-label', 'result-file']) {
    if (!values[required]) {
      throw new Error(`Missing --${required}`);
    }
  }
  return {
    modelId: values['model-id'],
    modelName: values['model-name'],
    dtype: values.dtype,
    variantLabel: values['variant-label'],
    modelFileName: values['model-file-name'] ?? null,
    backend: values.backend ?? 'cpu',
    maxTexts: positiveInteger(values['max-texts'], '--max-texts'),
    resultFile: values['result-file'],
  };
}

async function main() {
  const args = parseCli();
  const { documents, queryPairs } = await loadBenchmarkCorpus(args.maxTexts);
  const startedAt = performance.now();
  const monitor = new MemoryMonitor();
  monitor.start();

  const result = {
    model_id: args.modelId,
    model_name: args.modelName,
    variant: args.variantLabel,
    dtype: args.dtype,
    model_file_name: args.modelFileName,
    backend_requested: args.backend,
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
  const loadStartedAt = performance.now();
  try {
    const options = { dtype: args.dtype };
    if (args.modelFileName) {
      options.model_file_name = args.modelFileName;
    }
    extractor = await createFeatureExtractor(args.modelId, options, args.backend);
    result.backend_used = extractor._benchmark_backend ?? args.backend;
    result.load_time_ms = round(performance.now() - loadStartedAt);

    const latencies = [];
    const embeddings = new Map();
    for (const document of documents) {
      const inferenceStartedAt = performance.now();
      const tensor = await extractor(document.text, { pooling: 'mean', normalize: true });
      const vector = tensor.tolist()[0];
      latencies.push(performance.now() - inferenceStartedAt);
      embeddings.set(document.id, vector);
      result.embedding_dim ??= vector.length;
    }

    result.inference = summarizeTimings(latencies);
    result.quality = computeQuality(embeddings, documents, queryPairs);
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = error instanceof Error ? error.message : String(error);
    result.load_time_ms ??= round(performance.now() - loadStartedAt);
  } finally {
    await dispose(extractor);
    global.gc?.();
    result.memory = monitor.stop();
    result.total_time_ms = round(performance.now() - startedAt);
    result.finished_at = new Date().toISOString();
    await writeJson(args.resultFile, result);
  }
}

main().catch(async (error) => {
  const index = process.argv.indexOf('--result-file');
  const resultFile = index === -1 ? null : process.argv[index + 1];
  if (resultFile) {
    await writeJson(resultFile, {
      status: 'error',
      error: error instanceof Error ? error.message : String(error),
      finished_at: new Date().toISOString(),
    });
  } else {
    console.error(error);
  }
  process.exit(1);
});

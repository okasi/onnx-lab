#!/usr/bin/env node
import {
  MemoryMonitor,
  dispose,
  loadBenchmarkCorpus,
  positiveInteger,
  round,
  summarizeTimings,
} from '../lib/benchmark-support.mjs';
import { runBrowserHtml } from '../lib/browser-runtime.mjs';
import { computeQuality } from '../lib/metrics.mjs';
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';

const backend = process.argv[2];
const dtype = process.argv[3];
const maxTexts = positiveInteger(process.argv[4] ?? '1', 'maxTexts');
const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';

function browserHtml(documents) {
  const serialized = JSON.stringify(documents.map(({ id, text }) => ({ id, text })));
  return `<!DOCTYPE html><html><body><script type="module">
const docs = ${serialized};
try {
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  if (!navigator.gpu) throw new Error('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const shaderF16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: shaderF16 ? ['shader-f16'] : [] });
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  env.backends.onnx.webgpu.device = device;
  const loadStart = performance.now();
  const extractor = await pipeline('feature-extraction', '${MODEL_ID}', { dtype: '${dtype}', device: 'webgpu' });
  const loadMs = performance.now() - loadStart;
  const latencies = [];
  const vectors = {};
  for (const doc of docs) {
    const inferStart = performance.now();
    const output = await extractor(doc.text, { pooling: 'mean', normalize: true });
    latencies.push(performance.now() - inferStart);
    vectors[doc.id] = output.tolist()[0];
  }
  const totalMs = latencies.reduce((sum, value) => sum + value, 0);
  window.__RESULT__ = {
    status: 'ok',
    load_ms: Math.round(loadMs),
    infer_ms: Math.round(totalMs / latencies.length),
    infer_total_ms: Math.round(totalMs),
    dim: vectors[docs[0].id]?.length,
    shader_f16: shaderF16,
    vectors,
  };
} catch (error) {
  window.__RESULT__ = { status: 'error', error: error.message };
}
</script></body></html>`;
}

async function runCpu() {
  const { documents, queryPairs } = await loadBenchmarkCorpus(maxTexts);
  const result = { backend: 'cpu', dtype, documents: documents.length, status: 'pending' };
  const monitor = new MemoryMonitor();
  monitor.start();
  const startedAt = performance.now();
  let extractor;
  try {
    const loadStartedAt = performance.now();
    extractor = await createFeatureExtractor(MODEL_ID, { dtype }, 'cpu');
    result.load_ms = round(performance.now() - loadStartedAt);
    const latencies = [];
    const embeddings = new Map();
    for (const document of documents) {
      const inferenceStartedAt = performance.now();
      const tensor = await extractor(document.text, { pooling: 'mean', normalize: true });
      latencies.push(performance.now() - inferenceStartedAt);
      embeddings.set(document.id, tensor.tolist()[0]);
    }
    const timings = summarizeTimings(latencies);
    result.infer_ms = timings.mean_ms;
    result.infer_total_ms = timings.total_ms;
    result.embedding_dim = embeddings.values().next().value?.length;
    if (documents.length > 1) {
      const quality = computeQuality(embeddings, documents, queryPairs);
      result.quality = quality.composite_score;
      result.xling = quality.cross_lingual_pairs.mean_cosine;
      result.xl_r5 = quality.cross_lingual_recall_at_5;
      result.r5 = quality.recall_at_5;
    }
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = (error instanceof Error ? error.message : String(error)).slice(0, 400);
  } finally {
    await dispose(extractor);
    result.peak_rss_mb = monitor.stop().peak_rss_mb;
  }
  result.total_ms = round(performance.now() - startedAt);
  console.log(JSON.stringify(result));
}

async function runWebgpu() {
  const { documents, queryPairs } = await loadBenchmarkCorpus(maxTexts);
  const result = { backend: 'webgpu', dtype, documents: documents.length, status: 'pending' };
  const startedAt = performance.now();
  try {
    const { payload } = await runBrowserHtml(browserHtml(documents), {
      timeoutMs: 3_600_000,
      launchOptions: {
        args: ['--enable-unsafe-webgpu', '--use-angle=default'],
      },
    });
    if (payload.status !== 'ok') throw new Error(payload.error);
    result.load_ms = payload.load_ms;
    result.infer_ms = payload.infer_ms;
    result.infer_total_ms = payload.infer_total_ms;
    result.embedding_dim = payload.dim;
    result.shader_f16 = payload.shader_f16;
    if (documents.length > 1) {
      const quality = computeQuality(
        new Map(Object.entries(payload.vectors)),
        documents,
        queryPairs,
      );
      result.quality = quality.composite_score;
      result.xling = quality.cross_lingual_pairs.mean_cosine;
      result.xl_r5 = quality.cross_lingual_recall_at_5;
      result.r5 = quality.recall_at_5;
    }
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = (error instanceof Error ? error.message : String(error)).slice(0, 400);
  }
  result.total_ms = round(performance.now() - startedAt);
  console.log(JSON.stringify(result));
}

if (backend === 'cpu') {
  await runCpu();
} else if (backend === 'webgpu') {
  await runWebgpu();
} else {
  console.log(JSON.stringify({ status: 'error', error: `unknown backend ${backend}` }));
  process.exit(1);
}

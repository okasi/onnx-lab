#!/usr/bin/env node
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';
import { computeQuality } from '../lib/metrics.mjs';
import { cosineSimilarity } from '../lib/metrics.mjs';

const mode = process.argv[2];
const maxTexts = Number(process.argv[3] ?? 54);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const PORT = 8780;

function round(n, d = 2) {
  return Number(Number(n).toFixed(d));
}

function memSnapshot() {
  const m = process.memoryUsage();
  return {
    rss_mb: round(m.rss / 1024 / 1024),
    heap_used_mb: round(m.heapUsed / 1024 / 1024),
    external_mb: round(m.external / 1024 / 1024),
  };
}

async function loadCorpus() {
  const corpus = JSON.parse(
    await fs.readFile(path.join(root, 'data', 'benchmark-corpus.json'), 'utf8'),
  );
  const documents = corpus.documents.slice(0, maxTexts);
  const docIds = new Set(documents.map((d) => d.id));
  const queryPairs = corpus.query_pairs.filter(
    (p) => docIds.has(p.sv_doc_id) && docIds.has(p.tr_doc_id),
  );
  return { documents, queryPairs };
}

async function runCpu() {
  const { documents, queryPairs } = await loadCorpus();
  const result = { backend: 'cpu', status: 'pending', dtype: 'q4', documents: documents.length };
  const wallStart = performance.now();
  let peak = memSnapshot();

  const sample = () => {
    const s = memSnapshot();
    for (const k of Object.keys(peak)) {
      if (s[k] > peak[k]) peak[k] = s[k];
    }
  };

  try {
    const loadStart = performance.now();
    const extractor = await createFeatureExtractor(MODEL_ID, { dtype: 'q4' }, 'cpu');
    result.load_time_ms = round(performance.now() - loadStart);
    sample();

    const latencies = [];
    const embeddings = new Map();
    const cpuVectors = [];

    for (const doc of documents) {
      const t0 = performance.now();
      const tensor = await extractor(doc.text, { pooling: 'mean', normalize: true });
      latencies.push(performance.now() - t0);
      const vector = tensor.tolist()[0];
      embeddings.set(doc.id, vector);
      cpuVectors.push(vector);
      sample();
    }

    result.inference = {
      count: latencies.length,
      mean_ms: round(latencies.reduce((a, b) => a + b, 0) / latencies.length),
      total_ms: round(latencies.reduce((a, b) => a + b, 0)),
    };
    result.quality = computeQuality(embeddings, documents, queryPairs);
    result.memory = { peak_rss_mb: peak.rss_mb, peak_heap_used_mb: peak.heap_used_mb, peak_external_mb: peak.external_mb };
    result.embedding_dim = cpuVectors[0]?.length;
    result.status = 'ok';
    await extractor.dispose();
  } catch (e) {
    result.status = 'error';
    result.error = e.message;
  }

  result.total_time_ms = round(performance.now() - wallStart);
  // stash vectors for cross-check if needed
  result._vectors = undefined;
  console.log(JSON.stringify(result));
}

function browserHtml(documentsJson) {
  return `<!DOCTYPE html><html><body><script type="module">
const docs = ${documentsJson};
const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
env.allowRemoteModels = true;
env.useBrowserCache = true;
const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice();
env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
env.backends.onnx.webgpu.device = device;
const t0 = performance.now();
const extractor = await pipeline('feature-extraction', '${MODEL_ID}', { dtype: 'q4', device: 'webgpu' });
const loadMs = performance.now() - t0;
const latencies = [];
const vectors = {};
for (const doc of docs) {
  const t1 = performance.now();
  const out = await extractor(doc.text, { pooling: 'mean', normalize: true });
  latencies.push(performance.now() - t1);
  vectors[doc.id] = out.tolist()[0];
}
window.__RESULT__ = {
  status: 'ok',
  load_time_ms: Math.round(loadMs),
  inference: {
    count: latencies.length,
    mean_ms: Math.round(latencies.reduce((a,b)=>a+b,0)/latencies.length),
    total_ms: Math.round(latencies.reduce((a,b)=>a+b,0)),
  },
  vectors,
  shader_f16: adapter.features.has('shader-f16'),
  adapter: adapter.info?.description || adapter.info?.vendor || 'unknown',
};
</script></body></html>`;
}

async function runWebgpu() {
  const { documents, queryPairs } = await loadCorpus();
  const result = { backend: 'webgpu', status: 'pending', dtype: 'q4', documents: documents.length };

  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(browserHtml(JSON.stringify(documents.map((d) => ({ id: d.id, text: d.text })))));
  });
  await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

  const wallStart = performance.now();
  try {
    const browser = await chromium.launch({
      channel: 'chrome',
      headless: true,
      args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    });
    try {
      const page = await browser.newPage();
      page.setDefaultTimeout(3600000);
      await page.goto(`http://127.0.0.1:${PORT}/`);
      await page.waitForFunction(() => window.__RESULT__, null, { timeout: 3600000 });
      const payload = await page.evaluate(() => window.__RESULT__);

      const embeddings = new Map(Object.entries(payload.vectors));
      result.load_time_ms = payload.load_time_ms;
      result.inference = payload.inference;
      result.shader_f16 = payload.shader_f16;
      result.adapter = payload.adapter;
      result.quality = computeQuality(embeddings, documents, queryPairs);
      result.embedding_dim = embeddings.values().next().value?.length;
      result.status = 'ok';

      // Compare vs CPU on same machine
      const cpuExtractor = await createFeatureExtractor(MODEL_ID, { dtype: 'q4' }, 'cpu');
      let maxDiff = 0;
      for (const doc of documents.slice(0, Math.min(5, documents.length))) {
        const cpuOut = await cpuExtractor(doc.text, { pooling: 'mean', normalize: true });
        const cpuVec = cpuOut.tolist()[0];
        const gpuVec = embeddings.get(doc.id);
        const sim = cosineSimilarity(cpuVec, gpuVec);
        maxDiff = Math.max(maxDiff, 1 - sim);
      }
      await cpuExtractor.dispose();
      result.max_cosine_diff_vs_cpu = round(maxDiff, 6);

      const chromePid = browser.process()?.pid;
      if (chromePid) {
        try {
          const stat = await fs.readFile(`/proc/${chromePid}/status`, 'utf8');
          const m = stat.match(/VmRSS:\s+(\d+)/);
          if (m) result.chrome_rss_mb = round(Number(m[1]) / 1024);
        } catch { /* ignore */ }
      }
    } finally {
      await browser.close();
    }
  } catch (e) {
    result.status = 'error';
    result.error = e.message;
  } finally {
    server.close();
  }

  result.total_time_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

if (mode === 'cpu') await runCpu();
else if (mode === 'webgpu') await runWebgpu();
else {
  console.log(JSON.stringify({ status: 'error', error: `unknown mode ${mode}` }));
  process.exit(1);
}

#!/usr/bin/env node
/**
 * Test EmbeddingGemma quants on CPU and WebGPU (smoke: 1 doc, or --full N docs).
 * Usage: node probe-gemma-quant-worker.mjs <cpu|webgpu> <dtype> [maxTexts]
 */
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';
import { computeQuality } from '../lib/metrics.mjs';

const backend = process.argv[2];
const dtype = process.argv[3];
const maxTexts = Number(process.argv[4] ?? 1);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const PORT = 8781 + Math.floor(Math.random() * 100);

function round(n, d = 2) {
  return Number(Number(n).toFixed(d));
}

function memSnapshot() {
  const m = process.memoryUsage();
  return {
    rss_mb: round(m.rss / 1024 / 1024),
    heap_used_mb: round(m.heapUsed / 1024 / 1024),
  };
}

async function loadCorpus(n) {
  const corpus = JSON.parse(
    await fs.readFile(path.join(root, 'data', 'benchmark-corpus.json'), 'utf8'),
  );
  const documents = corpus.documents.slice(0, n);
  const docIds = new Set(documents.map((d) => d.id));
  const queryPairs = corpus.query_pairs.filter(
    (p) => docIds.has(p.sv_doc_id) && docIds.has(p.tr_doc_id),
  );
  return { documents, queryPairs };
}

async function runCpu() {
  const { documents, queryPairs } = await loadCorpus(maxTexts);
  const result = { backend: 'cpu', dtype, documents: documents.length, status: 'pending' };
  const wallStart = performance.now();
  let peak = memSnapshot();
  const sample = () => {
    const s = memSnapshot();
    for (const k of Object.keys(peak)) if (s[k] > peak[k]) peak[k] = s[k];
  };

  try {
    const loadStart = performance.now();
    const extractor = await createFeatureExtractor(MODEL_ID, { dtype }, 'cpu');
    result.load_ms = round(performance.now() - loadStart);
    sample();

    const latencies = [];
    const embeddings = new Map();
    for (const doc of documents) {
      const t0 = performance.now();
      const tensor = await extractor(doc.text, { pooling: 'mean', normalize: true });
      latencies.push(performance.now() - t0);
      embeddings.set(doc.id, tensor.tolist()[0]);
      sample();
    }

    result.infer_ms = round(latencies.reduce((a, b) => a + b, 0) / latencies.length);
    result.infer_total_ms = round(latencies.reduce((a, b) => a + b, 0));
    result.embedding_dim = embeddings.values().next().value?.length;
    if (maxTexts > 1) {
      const q = computeQuality(embeddings, documents, queryPairs);
      result.quality = q.composite_score;
      result.xling = q.cross_lingual_pairs?.mean_cosine;
      result.xl_r5 = q.cross_lingual_recall_at_5;
      result.r5 = q.recall_at_5;
    }
    result.peak_rss_mb = peak.rss_mb;
    result.status = 'ok';
    await extractor.dispose();
  } catch (e) {
    result.status = 'error';
    result.error = (e?.message ?? String(e)).slice(0, 400);
  }
  result.total_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

function browserHtml(dtype, documents) {
  const docsJson = JSON.stringify(documents.map((d) => ({ id: d.id, text: d.text })));
  return `<!DOCTYPE html><html><body><script type="module">
const docs = ${docsJson};
try {
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  if (!navigator.gpu) throw new Error('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const f16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: f16 ? ['shader-f16'] : [] });
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  env.backends.onnx.webgpu.device = device;
  const t0 = performance.now();
  const extractor = await pipeline('feature-extraction', '${MODEL_ID}', { dtype: '${dtype}', device: 'webgpu' });
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
    status: 'ok', load_ms: Math.round(loadMs),
    infer_ms: Math.round(latencies.reduce((a,b)=>a+b,0)/latencies.length),
    infer_total_ms: Math.round(latencies.reduce((a,b)=>a+b,0)),
    dim: vectors[docs[0].id]?.length, shader_f16: f16, vectors,
  };
} catch (e) {
  window.__RESULT__ = { status: 'error', error: e.message };
}
</script></body></html>`;
}

async function runWebgpu() {
  const { documents, queryPairs } = await loadCorpus(maxTexts);
  const result = { backend: 'webgpu', dtype, documents: documents.length, status: 'pending' };
  const wallStart = performance.now();

  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(browserHtml(dtype, documents));
  });
  await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

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
      if (payload.status !== 'ok') throw new Error(payload.error);

      result.load_ms = payload.load_ms;
      result.infer_ms = payload.infer_ms;
      result.infer_total_ms = payload.infer_total_ms;
      result.embedding_dim = payload.dim;
      result.shader_f16 = payload.shader_f16;

      if (maxTexts > 1 && payload.vectors) {
        const embeddings = new Map(Object.entries(payload.vectors));
        const q = computeQuality(embeddings, documents, queryPairs);
        result.quality = q.composite_score;
        result.xling = q.cross_lingual_pairs?.mean_cosine;
        result.xl_r5 = q.cross_lingual_recall_at_5;
        result.r5 = q.recall_at_5;
      }
      result.status = 'ok';
    } finally {
      await browser.close();
    }
  } catch (e) {
    result.status = 'error';
    result.error = (e?.message ?? String(e)).slice(0, 400);
  } finally {
    server.close();
  }
  result.total_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

if (backend === 'cpu') await runCpu();
else if (backend === 'webgpu') await runWebgpu();
else {
  console.log(JSON.stringify({ status: 'error', error: `unknown backend ${backend}` }));
  process.exit(1);
}

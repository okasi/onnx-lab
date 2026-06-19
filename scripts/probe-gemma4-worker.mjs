#!/usr/bin/env node
/**
 * Smoke-test one Gemma 4 ONNX LLM variant (load + short generation).
 * Usage: node probe-gemma4-worker.mjs <modelId> <cpu|wasm-jsep|wasm|webgpu> <dtype> [maxNewTokens]
 */
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import {
  GEMMA4_DEFAULT_MAX_NEW_TOKENS,
  GEMMA4_DEFAULT_PROMPT,
} from '../config/gemma4-models.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

const modelId = process.argv[2];
const backend = process.argv[3];
const dtype = process.argv[4];
const maxNewTokens = Number(process.argv[5] ?? GEMMA4_DEFAULT_MAX_NEW_TOKENS);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const PORT = 8791 + Math.floor(Math.random() * 100);

function round(n, d = 2) {
  return Number(Number(n).toFixed(d));
}

function memSnapshot() {
  const m = process.memoryUsage();
  return { rss_mb: round(m.rss / 1024 / 1024), heap_used_mb: round(m.heapUsed / 1024 / 1024) };
}

function classifyError(message) {
  const m = message.toLowerCase();
  if (m.includes('bad_alloc') || m.includes('out of memory') || m.includes('oom')) {
    return 'oom';
  }
  if (m.includes('gatherblockquantized') || m.includes('bits==4 or 8')) {
    return 'gather_quant';
  }
  if (m.includes('shader-f16') || m.includes('requires f16')) {
    return 'shader_f16';
  }
  return 'error';
}

async function runNodeBackend() {
  const result = {
    model_id: modelId,
    backend,
    dtype,
    max_new_tokens: maxNewTokens,
    status: 'pending',
    load_ms: null,
    infer_ms: null,
    generated_text: null,
    error: null,
    error_kind: null,
    peak_rss_mb: null,
    webgpu_dtype_fallback: null,
  };

  const wallStart = performance.now();
  let peak = memSnapshot();
  const sample = () => {
    const s = memSnapshot();
    for (const k of Object.keys(peak)) {
      if (s[k] > peak[k]) peak[k] = s[k];
    }
  };

  let generator;
  try {
    const loadStart = performance.now();
    generator = await createTextGenerator(modelId, { dtype }, backend);
    result.load_ms = round(performance.now() - loadStart);
    result.backend_used = generator._benchmark_backend ?? backend;
    result.webgpu_dtype_fallback = generator._webgpu_dtype_fallback ?? null;
    sample();

    const inferStart = performance.now();
    const outputs = await generator(GEMMA4_DEFAULT_PROMPT, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
    });
    result.infer_ms = round(performance.now() - inferStart);
    result.generated_text = outputs?.[0]?.generated_text ?? null;
    result.status = 'ok';
    sample();
  } catch (e) {
    const msg = (e?.message ?? String(e)).slice(0, 500);
    result.error = msg;
    result.error_kind = classifyError(msg);
    result.status = result.load_ms != null ? 'infer_error' : 'load_error';
  } finally {
    if (generator) {
      try {
        await generator.dispose();
      } catch {
        // ignore dispose errors after OOM
      }
    }
  }

  result.peak_rss_mb = peak.rss_mb;
  result.total_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

function browserHtml(model, quant, maxTokens) {
  const prompt = JSON.stringify(GEMMA4_DEFAULT_PROMPT);
  return `<!DOCTYPE html><html><body><script type="module">
const prompt = ${prompt};
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
  const loadStart = performance.now();
  let usedDtype = '${quant}';
  let generator;
  try {
    generator = await pipeline('text-generation', '${model}', { dtype: usedDtype, device: 'webgpu' });
  } catch (e) {
    const msg = e?.message ?? String(e);
    if ((msg.includes('shader-f16') || msg.includes('requires f16')) && usedDtype !== 'q4') {
      usedDtype = 'q4';
      generator = await pipeline('text-generation', '${model}', { dtype: usedDtype, device: 'webgpu' });
    } else {
      throw e;
    }
  }
  const loadMs = performance.now() - loadStart;
  const inferStart = performance.now();
  const out = await generator(prompt, { max_new_tokens: ${maxTokens}, do_sample: false });
  window.__RESULT__ = {
    status: 'ok',
    load_ms: Math.round(loadMs),
    infer_ms: Math.round(performance.now() - inferStart),
    generated_text: out?.[0]?.generated_text ?? null,
    shader_f16: f16,
    dtype_used: usedDtype,
  };
} catch (e) {
  window.__RESULT__ = { status: 'error', error: e.message };
}
</script></body></html>`;
}

async function runWebgpu() {
  const result = {
    model_id: modelId,
    backend: 'webgpu',
    dtype,
    max_new_tokens: maxNewTokens,
    status: 'pending',
    load_ms: null,
    infer_ms: null,
    generated_text: null,
    error: null,
    error_kind: null,
    shader_f16: null,
    dtype_used: dtype,
  };

  const wallStart = performance.now();
  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(browserHtml(modelId, dtype, maxNewTokens));
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
      page.setDefaultTimeout(3_600_000);
      await page.goto(`http://127.0.0.1:${PORT}/`);
      await page.waitForFunction(() => window.__RESULT__, null, { timeout: 3_600_000 });
      const payload = await page.evaluate(() => window.__RESULT__);
      if (payload.status !== 'ok') {
        throw new Error(payload.error);
      }
      result.load_ms = payload.load_ms;
      result.infer_ms = payload.infer_ms;
      result.generated_text = payload.generated_text;
      result.shader_f16 = payload.shader_f16;
      result.dtype_used = payload.dtype_used;
      if (payload.dtype_used && payload.dtype_used !== dtype) {
        result.webgpu_dtype_fallback = { from: dtype, to: payload.dtype_used };
      }
      result.status = 'ok';
    } finally {
      await browser.close();
    }
  } catch (e) {
    const msg = (e?.message ?? String(e)).slice(0, 500);
    result.error = msg;
    result.error_kind = classifyError(msg);
    result.status = result.load_ms != null ? 'infer_error' : 'load_error';
  } finally {
    server.close();
  }

  result.total_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

if (!modelId || !backend || !dtype) {
  console.log(JSON.stringify({ status: 'load_error', error: 'usage: modelId backend dtype [maxNewTokens]' }));
  process.exit(1);
}

if (backend === 'webgpu') {
  await runWebgpu();
} else if (['cpu', 'wasm', 'wasm-jsep'].includes(backend)) {
  await runNodeBackend();
} else {
  console.log(JSON.stringify({ status: 'load_error', error: `unknown backend ${backend}` }));
  process.exit(1);
}

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
import {
  GEMMA4_WEBGPU_STRATEGIES,
  classifyGemma4Error,
  gemma4BrowserHtml,
  round,
} from '../lib/gemma4-helpers.mjs';
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
  return classifyGemma4Error(message);
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

async function runWebgpu() {
  const hard = process.argv[6] === '--hard';
  const strategies = hard
    ? ['browser:q4-control', 'browser:q4-force-gather', 'browser:q4-dual-ep', 'browser:q8', 'browser:angle-gl']
    : ['browser:q4-control'];

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
    strategies_tried: [],
  };

  const wallStart = performance.now();

  for (const strategyName of strategies) {
    const spec = GEMMA4_WEBGPU_STRATEGIES[strategyName];
    if (!spec) continue;

    const attempt = { strategy: strategyName, status: 'pending' };
    result.strategies_tried.push(attempt);

    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(gemma4BrowserHtml(modelId, { ...spec, dtype }, GEMMA4_DEFAULT_PROMPT, maxNewTokens));
    });
    await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

    try {
      const browser = await chromium.launch({
        channel: 'chrome',
        headless: true,
        args: spec.args ?? ['--enable-unsafe-webgpu', '--use-angle=default'],
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
        result.webgpu_strategy = strategyName;
        if (payload.dtype_used && payload.dtype_used !== dtype) {
          result.webgpu_dtype_fallback = { from: dtype, to: payload.dtype_used };
        }
        result.status = 'ok';
        attempt.status = 'ok';
        await browser.close();
        server.close();
        break;
      } catch (e) {
        await browser.close();
        const msg = (e?.message ?? String(e)).slice(0, 300);
        attempt.status = 'error';
        attempt.error = msg;
        result.error = msg;
        result.error_kind = classifyError(msg);
        result.status = result.load_ms != null ? 'infer_error' : 'load_error';
      }
    } catch (e) {
      const msg = (e?.message ?? String(e)).slice(0, 300);
      attempt.status = 'error';
      attempt.error = msg;
      result.error = msg;
      result.error_kind = classifyError(msg);
      result.status = 'load_error';
    } finally {
      server.close();
    }

    if (result.status === 'ok') break;
    if (!hard) break;
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

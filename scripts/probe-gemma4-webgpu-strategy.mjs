#!/usr/bin/env node
/**
 * Run one Gemma 4 WebGPU browser strategy.
 * Usage: node probe-gemma4-webgpu-strategy.mjs <modelId> <strategyName>
 */
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { GEMMA4_DEFAULT_MAX_NEW_TOKENS, GEMMA4_DEFAULT_PROMPT } from '../config/gemma4-models.mjs';
import {
  GEMMA4_WEBGPU_STRATEGIES,
  classifyGemma4Error,
  gemma4BrowserHtml,
  round,
} from '../lib/gemma4-helpers.mjs';

const modelId = process.argv[2];
const strategy = process.argv[3];
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const PORT = 8781 + Math.floor(Math.random() * 100);

async function main() {
  const spec = GEMMA4_WEBGPU_STRATEGIES[strategy];
  const result = {
    model_id: modelId,
    strategy,
    status: 'pending',
    load_ms: null,
    infer_ms: null,
    generated_text: null,
    shader_f16: null,
    dtype_used: spec?.dtype ?? null,
    error: null,
    error_kind: null,
  };

  if (!modelId || !spec) {
    result.status = 'error';
    result.error = !modelId ? 'model id required' : `unknown strategy: ${strategy}`;
    console.log(JSON.stringify(result));
    return;
  }

  const server = http.createServer((_req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(gemma4BrowserHtml(modelId, spec, GEMMA4_DEFAULT_PROMPT, GEMMA4_DEFAULT_MAX_NEW_TOKENS));
  });
  await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

  const wallStart = performance.now();
  try {
    const browser = await chromium.launch({
      channel: 'chrome',
      headless: spec.headless ?? true,
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
      result.status = 'ok';
    } finally {
      await browser.close();
    }
  } catch (e) {
    const msg = (e?.message ?? String(e)).slice(0, 500);
    result.error = msg;
    result.error_kind = classifyGemma4Error(msg);
    result.status = result.load_ms != null ? 'infer_error' : 'error';
  } finally {
    server.close();
  }

  result.total_ms = round(performance.now() - wallStart);
  console.log(JSON.stringify(result));
}

main().catch((e) => {
  console.log(JSON.stringify({ status: 'error', error: e.message }));
  process.exit(1);
});

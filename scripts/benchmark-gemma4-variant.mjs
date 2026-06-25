#!/usr/bin/env node
/**
 * Benchmark one Gemma 4 model × quant × backend in an isolated process.
 */
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';
import { GEMMA4_DEFAULT_MAX_NEW_TOKENS } from '../config/gemma4-models.mjs';
import {
  MemoryMonitor,
  classifyGemma4Error,
  gemma4BrowserHtml,
  GEMMA4_WEBGPU_STRATEGIES,
  round,
  summarizeTimings,
} from '../lib/gemma4-helpers.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--model-id') args.modelId = argv[++i];
    else if (key === '--model-name') args.modelName = argv[++i];
    else if (key === '--model-slug') args.modelSlug = argv[++i];
    else if (key === '--dtype') args.dtype = argv[++i];
    else if (key === '--backend') args.backend = argv[++i];
    else if (key === '--max-prompts') args.maxPrompts = Number(argv[++i]);
    else if (key === '--max-new-tokens') args.maxNewTokens = Number(argv[++i]);
    else if (key === '--webgpu-strategy') args.webgpuStrategy = argv[++i];
    else if (key === '--result-file') args.resultFile = argv[++i];
  }
  return args;
}

async function loadPrompts(maxPrompts) {
  const raw = await fs.readFile(path.join(root, 'data', 'gemma4-benchmark-prompts.json'), 'utf8');
  const data = JSON.parse(raw);
  return data.prompts.slice(0, maxPrompts ?? data.prompts.length);
}

async function runNodeBenchmark(args, prompts, result) {
  const maxNewTokens = args.maxNewTokens ?? GEMMA4_DEFAULT_MAX_NEW_TOKENS;
  let generator;

  try {
    const loadStart = performance.now();
    generator = await createTextGenerator(
      args.modelId,
      {
        dtype: args.dtype,
        session_options: {
          enableCpuMemArena: false,
          enableMemPattern: false,
        },
      },
      args.backend,
    );
    result.load_time_ms = round(performance.now() - loadStart);
    result.backend_used = generator._benchmark_backend ?? args.backend;
    result.webgpu_dtype_fallback = generator._webgpu_dtype_fallback ?? null;
    result.load_status = 'ok';

    if (global.gc) global.gc();

    const latencies = [];
    const outputs = [];
    for (const item of prompts) {
      const t0 = performance.now();
      const out = await generator(item.prompt, {
        max_new_tokens: maxNewTokens,
        do_sample: false,
      });
      latencies.push(performance.now() - t0);
      outputs.push({
        id: item.id,
        language: item.language,
        topic: item.topic,
        generated_text: out?.[0]?.generated_text ?? null,
      });
    }

    result.inference = summarizeTimings(latencies);
    result.outputs = outputs;
    result.tokens_generated = maxNewTokens * prompts.length;
    result.tokens_per_sec = result.inference.total_ms > 0
      ? round((result.tokens_generated * 1000) / result.inference.total_ms, 2)
      : 0;
    result.status = 'ok';
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    result.error = msg.slice(0, 600);
    result.error_kind = classifyGemma4Error(msg);
    if (result.load_time_ms != null) {
      result.load_status = 'ok';
      result.status = 'infer_error';
    } else {
      result.load_status = 'error';
      result.status = 'error';
    }
  } finally {
    if (generator) {
      try {
        await generator.dispose();
      } catch {
        // ignore post-OOM dispose failures
      }
    }
  }
}

async function runWebgpuBenchmark(args, prompts, result) {
  const maxNewTokens = args.maxNewTokens ?? GEMMA4_DEFAULT_MAX_NEW_TOKENS;
  const strategyNames = args.webgpuStrategy
    ? [args.webgpuStrategy]
    : ['browser:q4-control', 'browser:q4-force-gather', 'browser:q4-dual-ep', 'browser:q8'];

  const prompt = prompts[0]?.prompt ?? 'Hello';
  result.strategies_tried = [];

  for (const strategyName of strategyNames) {
    const spec = GEMMA4_WEBGPU_STRATEGIES[strategyName];
    if (!spec) continue;

    const attempt = { strategy: strategyName, status: 'pending' };
    result.strategies_tried.push(attempt);

    const PORT = 8791 + Math.floor(Math.random() * 100);
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(gemma4BrowserHtml(args.modelId, { ...spec, dtype: args.dtype ?? spec.dtype }, prompt, maxNewTokens));
    });
    await new Promise((r) => server.listen(PORT, '127.0.0.1', r));

    try {
      const browser = await chromium.launch({
        channel: 'chrome',
        headless: spec.headless ?? true,
        args: spec.args ?? ['--enable-unsafe-webgpu', '--use-angle=default'],
      });
      try {
        const page = await browser.newPage();
        page.setDefaultTimeout(3_600_000);
        const wallStart = performance.now();
        await page.goto(`http://127.0.0.1:${PORT}/`);
        await page.waitForFunction(() => window.__RESULT__, null, { timeout: 3_600_000 });
        const payload = await page.evaluate(() => window.__RESULT__);
        result.total_time_ms = round(performance.now() - wallStart);

        if (payload.status !== 'ok') {
          throw new Error(payload.error);
        }

        result.load_time_ms = payload.load_ms;
        result.load_status = 'ok';
        result.inference = {
          count: 1,
          mean_ms: payload.infer_ms,
          total_ms: payload.infer_ms,
          p50_ms: payload.infer_ms,
          p95_ms: payload.infer_ms,
        };
        result.outputs = [{
          id: prompts[0]?.id ?? 'prompt-0',
          language: prompts[0]?.language,
          topic: prompts[0]?.topic,
          generated_text: payload.generated_text,
        }];
        result.tokens_generated = maxNewTokens;
        result.tokens_per_sec = payload.infer_ms > 0
          ? round((maxNewTokens * 1000) / payload.infer_ms, 2)
          : 0;
        result.shader_f16 = payload.shader_f16;
        result.dtype_used = payload.dtype_used;
        result.webgpu_strategy = strategyName;
        if (payload.dtype_used && payload.dtype_used !== args.dtype) {
          result.webgpu_dtype_fallback = { from: args.dtype, to: payload.dtype_used };
        }
        result.backend_used = 'webgpu';
        result.status = 'ok';
        attempt.status = 'ok';
        await browser.close();
        server.close();
        break;
      } catch (inner) {
        await browser.close();
        throw inner;
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      attempt.status = 'error';
      attempt.error = msg.slice(0, 300);
      result.error = msg.slice(0, 600);
      result.error_kind = classifyGemma4Error(msg);
      result.status = result.load_time_ms != null ? 'infer_error' : 'error';
      result.load_status = result.load_time_ms != null ? 'ok' : 'error';
      result.webgpu_strategy = strategyName;
    } finally {
      server.close();
    }

    if (result.status === 'ok') break;
  }

  if (prompts.length > 1 && result.status === 'ok') {
    result.note = 'webgpu benchmark uses first prompt only; CPU/wasm run all prompts';
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const prompts = await loadPrompts(args.maxPrompts);
  const mem = new MemoryMonitor();
  mem.start();

  const wallStart = performance.now();
  const result = {
    model_id: args.modelId,
    model_name: args.modelName,
    model_slug: args.modelSlug,
    dtype: args.dtype,
    backend_requested: args.backend,
    backend_used: null,
    status: 'pending',
    load_status: null,
    started_at: new Date().toISOString(),
    load_time_ms: null,
    total_time_ms: null,
    max_new_tokens: args.maxNewTokens ?? GEMMA4_DEFAULT_MAX_NEW_TOKENS,
    prompt_count: prompts.length,
    inference: null,
    tokens_generated: null,
    tokens_per_sec: null,
    outputs: null,
    memory: null,
    error: null,
    error_kind: null,
  };

  if (args.backend === 'webgpu') {
    await runWebgpuBenchmark(args, prompts, result);
  } else {
    await runNodeBenchmark(args, prompts, result);
  }

  if (global.gc) global.gc();
  result.memory = mem.stop();
  result.total_time_ms ??= round(performance.now() - wallStart);
  result.finished_at = new Date().toISOString();

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

#!/usr/bin/env node
import {
  GEMMA4_DEFAULT_MAX_NEW_TOKENS,
  GEMMA4_DEFAULT_PROMPT,
} from '../config/gemma4-models.mjs';
import {
  MemoryMonitor,
  classifyRuntimeError,
  dispose,
  positiveInteger,
  round,
} from '../lib/benchmark-support.mjs';
import {
  runGemma4BrowserStrategy,
} from '../lib/gemma4-webgpu.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

const [modelId, backend, dtype] = process.argv.slice(2, 5);
const maxNewTokensArg = process.argv.slice(5).find((value) => !value.startsWith('--'));
const maxNewTokens = positiveInteger(
  maxNewTokensArg ?? String(GEMMA4_DEFAULT_MAX_NEW_TOKENS),
  'maxNewTokens',
);
const hard = process.argv.includes('--hard');

async function runNodeBackend() {
  const monitor = new MemoryMonitor();
  monitor.start();
  const startedAt = performance.now();
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
  let generator;
  try {
    const loadStartedAt = performance.now();
    generator = await createTextGenerator(modelId, { dtype }, backend);
    result.load_ms = round(performance.now() - loadStartedAt);
    result.backend_used = generator._benchmark_backend ?? backend;
    result.webgpu_dtype_fallback = generator._webgpu_dtype_fallback ?? null;

    const inferStartedAt = performance.now();
    const output = await generator(GEMMA4_DEFAULT_PROMPT, {
      max_new_tokens: maxNewTokens,
      do_sample: false,
    });
    result.infer_ms = round(performance.now() - inferStartedAt);
    result.generated_text = output?.[0]?.generated_text ?? null;
    result.status = 'ok';
  } catch (error) {
    const message = (error instanceof Error ? error.message : String(error)).slice(0, 500);
    result.error = message;
    result.error_kind = classifyRuntimeError(message);
    result.status = result.load_ms == null ? 'load_error' : 'infer_error';
  } finally {
    await dispose(generator);
    result.peak_rss_mb = monitor.stop().peak_rss_mb;
  }
  result.total_ms = round(performance.now() - startedAt);
  console.log(JSON.stringify(result));
}

async function runWebgpu() {
  const strategies = hard
    ? ['browser:q4-control', 'browser:q4-force-gather', 'browser:q4-dual-ep', 'browser:angle-gl']
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
  const startedAt = performance.now();

  for (const strategyName of strategies) {
    const attempt = { strategy: strategyName, status: 'pending' };
    result.strategies_tried.push(attempt);
    try {
      const payload = await runGemma4BrowserStrategy({
        modelId,
        strategyName,
        prompts: [{ id: 'smoke', prompt: GEMMA4_DEFAULT_PROMPT }],
        maxNewTokens,
        dtype,
      });
      result.load_ms = payload.load_ms;
      result.infer_ms = payload.inference.mean_ms;
      result.generated_text = payload.outputs[0]?.generated_text ?? null;
      result.shader_f16 = payload.shader_f16;
      result.dtype_used = payload.dtype_used;
      result.webgpu_strategy = strategyName;
      if (payload.dtype_used !== dtype) {
        result.webgpu_dtype_fallback = { from: dtype, to: payload.dtype_used };
      }
      result.status = 'ok';
      attempt.status = 'ok';
      break;
    } catch (error) {
      const message = (error instanceof Error ? error.message : String(error)).slice(0, 300);
      attempt.status = 'error';
      attempt.error = message;
      result.error = message;
      result.error_kind = classifyRuntimeError(message);
      result.status = result.load_ms == null ? 'load_error' : 'infer_error';
    }
  }
  result.total_ms = round(performance.now() - startedAt);
  console.log(JSON.stringify(result));
}

if (!modelId || !backend || !dtype) {
  console.log(JSON.stringify({
    status: 'load_error',
    error: 'usage: modelId backend dtype [maxNewTokens] [--hard]',
  }));
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

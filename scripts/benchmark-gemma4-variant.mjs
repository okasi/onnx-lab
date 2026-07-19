#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { GEMMA4_DEFAULT_MAX_NEW_TOKENS } from '../config/gemma4-models.mjs';
import {
  MemoryMonitor,
  classifyRuntimeError,
  dispose,
  positiveInteger,
  projectPath,
  readJson,
  round,
  summarizeTimings,
  writeJson,
} from '../lib/benchmark-support.mjs';
import {
  GEMMA4_WEBGPU_STRATEGIES,
  runGemma4BrowserStrategy,
} from '../lib/gemma4-webgpu.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

function parseCli() {
  const { values } = parseArgs({
    options: {
      'model-id': { type: 'string' },
      'model-name': { type: 'string' },
      'model-slug': { type: 'string' },
      dtype: { type: 'string' },
      backend: { type: 'string' },
      'max-prompts': { type: 'string' },
      'max-new-tokens': { type: 'string' },
      'webgpu-strategy': { type: 'string' },
      'result-file': { type: 'string' },
    },
    strict: true,
  });
  for (const required of ['model-id', 'model-slug', 'dtype', 'backend', 'result-file']) {
    if (!values[required]) throw new Error(`Missing --${required}`);
  }
  return {
    modelId: values['model-id'],
    modelName: values['model-name'] ?? values['model-slug'],
    modelSlug: values['model-slug'],
    dtype: values.dtype,
    backend: values.backend,
    maxPrompts: positiveInteger(values['max-prompts'], '--max-prompts'),
    maxNewTokens: positiveInteger(
      values['max-new-tokens'] ?? String(GEMMA4_DEFAULT_MAX_NEW_TOKENS),
      '--max-new-tokens',
    ),
    webgpuStrategy: values['webgpu-strategy'] ?? null,
    resultFile: values['result-file'],
  };
}

async function loadPrompts(maxPrompts) {
  const data = await readJson(projectPath('data', 'gemma4-benchmark-prompts.json'));
  return data.prompts.slice(0, maxPrompts ?? data.prompts.length);
}

async function runNodeBenchmark(args, prompts, result) {
  let generator;
  try {
    const loadStartedAt = performance.now();
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
    result.load_time_ms = round(performance.now() - loadStartedAt);
    result.backend_used = generator._benchmark_backend ?? args.backend;
    result.webgpu_dtype_fallback = generator._webgpu_dtype_fallback ?? null;
    result.load_status = 'ok';
    global.gc?.();

    const latencies = [];
    result.outputs = [];
    for (const item of prompts) {
      const startedAt = performance.now();
      const output = await generator(item.prompt, {
        max_new_tokens: args.maxNewTokens,
        do_sample: false,
      });
      latencies.push(performance.now() - startedAt);
      result.outputs.push({
        id: item.id,
        language: item.language,
        topic: item.topic,
        generated_text: output?.[0]?.generated_text ?? null,
      });
    }
    result.inference = summarizeTimings(latencies);
    result.status = 'ok';
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    result.error = message.slice(0, 600);
    result.error_kind = classifyRuntimeError(message);
    result.load_status = result.load_time_ms == null ? 'error' : 'ok';
    result.status = result.load_time_ms == null ? 'error' : 'infer_error';
  } finally {
    await dispose(generator);
  }
}

async function runWebgpuBenchmark(args, prompts, result) {
  const strategyNames = args.webgpuStrategy
    ? [args.webgpuStrategy]
    : ['browser:q4-control', 'browser:q4-force-gather', 'browser:q4-dual-ep'];
  result.strategies_tried = [];

  for (const strategyName of strategyNames) {
    if (!GEMMA4_WEBGPU_STRATEGIES[strategyName]) continue;
    const attempt = { strategy: strategyName, status: 'pending' };
    result.strategies_tried.push(attempt);
    try {
      const payload = await runGemma4BrowserStrategy({
        modelId: args.modelId,
        strategyName,
        prompts,
        maxNewTokens: args.maxNewTokens,
        dtype: args.dtype,
      });
      result.load_time_ms = payload.load_ms;
      result.load_status = 'ok';
      result.inference = payload.inference;
      result.outputs = payload.outputs;
      result.shader_f16 = payload.shader_f16;
      result.dtype_used = payload.dtype_used;
      result.webgpu_strategy = strategyName;
      result.backend_used = 'webgpu';
      result.total_time_ms = payload.total_ms;
      if (payload.dtype_used !== args.dtype) {
        result.webgpu_dtype_fallback = { from: args.dtype, to: payload.dtype_used };
      }
      result.status = 'ok';
      attempt.status = 'ok';
      break;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      attempt.status = 'error';
      attempt.error = message.slice(0, 300);
      result.error = message.slice(0, 600);
      result.error_kind = classifyRuntimeError(message);
      result.webgpu_strategy = strategyName;
      result.load_status = result.load_time_ms == null ? 'error' : 'ok';
      result.status = result.load_time_ms == null ? 'error' : 'infer_error';
    }
  }
}

async function main() {
  const args = parseCli();
  const prompts = await loadPrompts(args.maxPrompts);
  const monitor = new MemoryMonitor();
  monitor.start();
  const startedAt = performance.now();
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
    max_new_tokens: args.maxNewTokens,
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

  result.tokens_generated = result.status === 'ok'
    ? args.maxNewTokens * (result.inference?.count ?? 0)
    : null;
  result.tokens_per_sec = result.inference?.total_ms > 0
    ? round((result.tokens_generated * 1000) / result.inference.total_ms)
    : null;
  global.gc?.();
  result.memory = monitor.stop();
  result.total_time_ms ??= round(performance.now() - startedAt);
  result.finished_at = new Date().toISOString();
  await writeJson(args.resultFile, result);
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

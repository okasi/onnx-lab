#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  GEMMA4_BACKENDS,
  GEMMA4_DEFAULT_MAX_NEW_TOKENS,
  GEMMA4_MODELS,
  findGemma4Model,
} from '../config/gemma4-models.mjs';
import {
  RESULTS_DIR,
  ROOT_DIR,
  SCRIPTS_DIR,
  formatDuration,
  parseCsv,
  positiveInteger,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';
import { GEMMA4_WEBGPU_STRATEGIES } from '../lib/gemma4-webgpu.mjs';

const variantScript = path.join(SCRIPTS_DIR, 'benchmark-gemma4-variant.mjs');

function parseCli() {
  const { values } = parseArgs({
    options: {
      quick: { type: 'boolean' },
      'max-prompts': { type: 'string' },
      'max-new-tokens': { type: 'string' },
      model: { type: 'string' },
      dtype: { type: 'string' },
      backend: { type: 'string' },
      'webgpu-strategy': { type: 'string' },
      'include-fp32': { type: 'boolean' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;

  const quick = values.quick ?? false;
  const models = parseCsv(values.model) ?? (quick ? ['E2B-it'] : null);
  const dtypes = parseCsv(values.dtype) ?? (quick ? ['q4', 'q8'] : null);
  const backends = parseCsv(values.backend) ?? (quick ? ['cpu', 'webgpu'] : null);
  const unknownBackends = (backends ?? []).filter((backend) => !GEMMA4_BACKENDS.includes(backend));
  if (unknownBackends.length) {
    throw new Error(`Unknown backend(s): ${unknownBackends.join(', ')}`);
  }
  if (
    values['webgpu-strategy']
    && !GEMMA4_WEBGPU_STRATEGIES[values['webgpu-strategy']]
  ) {
    throw new Error(`Unknown WebGPU strategy: ${values['webgpu-strategy']}`);
  }

  return {
    quick,
    models,
    dtypes,
    backends,
    maxPrompts: positiveInteger(
      values['max-prompts'] ?? (quick ? '3' : null),
      '--max-prompts',
    ),
    maxNewTokens: positiveInteger(
      values['max-new-tokens'] ?? String(GEMMA4_DEFAULT_MAX_NEW_TOKENS),
      '--max-new-tokens',
    ),
    webgpuStrategy: values['webgpu-strategy'] ?? null,
    includeFp32: values['include-fp32'] ?? false,
    output: values.output ? path.resolve(ROOT_DIR, values.output) : null,
  };
}

function printHelp() {
  console.log(`Usage: node scripts/benchmark-gemma4.mjs [options]

Options:
  --quick              E2B q4+q8 on cpu+webgpu, 3 prompts
  --max-prompts N      Limit benchmark prompts
  --max-new-tokens N   Tokens per generation (default 8)
  --model slug[,slug]  Filter models (e.g. E2B-it,E4B-it)
  --dtype d[,d]        Filter quants
  --backend b[,b]      cpu | wasm-jsep | wasm | webgpu
  --webgpu-strategy S  Restrict WebGPU to one browser strategy
  --include-fp32       Include fp32
  --output path        Results JSON path
  -h, --help           Show this help
`);
}

function buildCells(args) {
  const models = args.models
    ? args.models.map((slug) => {
        const model = findGemma4Model(slug);
        if (!model) throw new Error(`Unknown model: ${slug}`);
        return model;
      })
    : GEMMA4_MODELS;
  const backends = args.backends ?? GEMMA4_BACKENDS;
  const quantOrder = ['q4', 'q8', 'q4f16', 'fp16', 'q2f16', 'fp32'];
  const cells = [];

  for (const model of models) {
    let quants = (args.dtypes ?? model.quants)
      .filter((dtype) => model.quants.includes(dtype));
    if (!args.includeFp32) {
      quants = quants.filter((dtype) => dtype !== 'fp32');
    }
    quants.sort((a, b) => {
      const left = quantOrder.indexOf(a);
      const right = quantOrder.indexOf(b);
      return (left === -1 ? 99 : left) - (right === -1 ? 99 : right);
    });
    for (const dtype of quants) {
      for (const backend of backends) {
        cells.push({ model, dtype, backend });
      }
    }
  }

  if (!cells.length) {
    throw new Error('No compatible model, dtype, and backend combinations selected');
  }
  return cells;
}

async function benchmarkCell({ model, dtype, backend }, args) {
  const workerArgs = [
    '--model-id', model.id,
    '--model-name', model.name,
    '--model-slug', model.slug,
    '--dtype', dtype,
    '--backend', backend,
    '--max-new-tokens', String(args.maxNewTokens),
  ];
  if (args.maxPrompts) {
    workerArgs.push('--max-prompts', String(args.maxPrompts));
  }
  if (args.webgpuStrategy && backend === 'webgpu') {
    workerArgs.push('--webgpu-strategy', args.webgpuStrategy);
  }
  const { result } = await runJsonWorker(variantScript, workerArgs, { resultFile: true });
  return {
    model_id: model.id,
    model_slug: model.slug,
    model_name: model.name,
    dtype,
    backend_requested: backend,
    ...result,
  };
}

function printSummary(run) {
  console.log(`\n${'='.repeat(120)}`);
  console.log('GEMMA 4 BENCHMARK SUMMARY');
  console.log('='.repeat(120));
  console.log(
    [
      'Model'.padEnd(18),
      'Quant'.padEnd(8),
      'Backend'.padEnd(10),
      'Status'.padEnd(12),
      'Load'.padStart(8),
      'ms/prompt'.padStart(10),
      'tok/s'.padStart(8),
      'RSS MB'.padStart(8),
    ].join(' '),
  );
  console.log('-'.repeat(120));
  for (const result of run.results) {
    console.log(
      [
        String(result.model_slug ?? result.model_name ?? '').slice(0, 17).padEnd(18),
        String(result.dtype).padEnd(8),
        String(result.backend_used ?? result.backend_requested ?? '-').padEnd(10),
        String(result.status).padEnd(12),
        (result.load_time_ms != null ? `${Math.round(result.load_time_ms)}ms` : '-').padStart(8),
        String(result.inference?.mean_ms ?? '-').padStart(10),
        String(result.tokens_per_sec ?? '-').padStart(8),
        String(result.memory?.peak_rss_mb ?? '-').padStart(8),
      ].join(' '),
    );
  }
  console.log('-'.repeat(120));
  console.log(`Wall: ${formatDuration(run.wall_time_ms)} | ok: ${run.summary.ok}/${run.summary.total}`);
  console.log('='.repeat(120));
}

async function main() {
  const args = parseCli();
  if (!args) {
    printHelp();
    return;
  }
  const cells = buildCells(args);
  console.log(
    `Gemma 4 benchmark - ${cells.length} variant(s), `
    + `${args.maxPrompts ?? 'all'} prompt(s), ${args.maxNewTokens} new tokens\n`,
  );

  const startedAt = performance.now();
  const results = [];
  for (const cell of cells) {
    const label = `${cell.model.slug} ${cell.dtype} ${cell.backend}`;
    process.stdout.write(`${label.padEnd(28)} ... `);
    const result = await benchmarkCell(cell, args);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`ok ${result.inference?.mean_ms}ms/prompt ${result.tokens_per_sec} tok/s`);
    } else {
      console.log(`${result.status}: ${(result.error ?? '').slice(0, 70)}`);
    }
  }

  const run = {
    benchmark: 'gemma4',
    tested_at: new Date().toISOString(),
    args: {
      max_prompts: args.maxPrompts,
      max_new_tokens: args.maxNewTokens,
      webgpu_strategy: args.webgpuStrategy,
    },
    wall_time_ms: Math.round(performance.now() - startedAt),
    results,
    summary: {
      total: results.length,
      ok: results.filter((result) => result.status === 'ok').length,
      infer_error: results.filter((result) => result.status === 'infer_error').length,
      error: results.filter((result) =>
        !['ok', 'infer_error'].includes(result.status)).length,
    },
  };

  const outPath = args.output ?? path.join(RESULTS_DIR, `benchmark-gemma4-${Date.now()}.json`);
  await writeJson(outPath, run);
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

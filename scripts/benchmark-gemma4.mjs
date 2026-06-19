#!/usr/bin/env node
/**
 * Gemma 4 ONNX LLM benchmark — models × quants × backends with isolated subprocesses.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import {
  GEMMA4_BACKENDS,
  GEMMA4_MODELS,
  findGemma4Model,
} from '../config/gemma4-models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const variantScript = path.join(__dirname, 'benchmark-gemma4-variant.mjs');

function parseArgs(argv) {
  const args = {
    quick: false,
    models: null,
    dtypes: null,
    backends: null,
    maxPrompts: null,
    maxNewTokens: 8,
    output: null,
    isolated: true,
    webgpuStrategy: null,
    includeFp32: false,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--quick') {
      args.quick = true;
      args.models = ['E2B-it'];
      args.dtypes = ['q4', 'q8'];
      args.backends = ['cpu', 'webgpu'];
      args.maxPrompts = 3;
      args.maxNewTokens = 8;
    } else if (arg === '--max-prompts') {
      args.maxPrompts = Number(argv[++i]);
    } else if (arg === '--max-new-tokens') {
      args.maxNewTokens = Number(argv[++i]);
    } else if (arg === '--model') {
      args.models = argv[++i].split(',');
    } else if (arg === '--dtype') {
      args.dtypes = argv[++i].split(',');
    } else if (arg === '--backend') {
      args.backends = argv[++i].split(',');
    } else if (arg === '--webgpu-strategy') {
      args.webgpuStrategy = argv[++i];
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--include-fp32') {
      args.includeFp32 = true;
    } else if (arg === '--no-isolate') {
      args.isolated = false;
    } else if (arg === '--help') {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/benchmark-gemma4.mjs [options]

Options:
  --quick              E2B q4+q8 on cpu+webgpu, 3 prompts
  --max-prompts N      Limit benchmark prompts
  --max-new-tokens N   Tokens per generation (default 8)
  --model slug[,slug]  Filter models (e.g. E2B-it,E4B-it)
  --dtype d[,d]        Filter quants
  --backend b[,b]       cpu | wasm-jsep | wasm | webgpu
  --webgpu-strategy    browser:q4-control (default), browser:q4-force-gather, ...
  --include-fp32       Include fp32 (multi-GB shards; skipped by default)
  --output path        Results JSON path
  --no-isolate         Run variant script in-process
`);
}

function formatDuration(ms) {
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function buildCells(args) {
  let models = GEMMA4_MODELS;
  if (args.models) {
    models = args.models.map((slug) => {
      const m = findGemma4Model(slug.trim());
      if (!m) throw new Error(`Unknown model: ${slug}`);
      return m;
    });
  }

  const backends = args.backends ?? GEMMA4_BACKENDS;
  const quantOrder = ['q4', 'q8', 'q4f16', 'fp16', 'q2f16', 'fp32'];
  const cells = [];
  for (const model of models) {
    let quants = args.dtypes ?? model.quants;
    if (!args.includeFp32) {
      quants = quants.filter((d) => d !== 'fp32');
    }
    quants = [...quants].sort((a, b) => {
      const ia = quantOrder.indexOf(a);
      const ib = quantOrder.indexOf(b);
      return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
    });
    for (const dtype of quants) {
      for (const backend of backends) {
        cells.push({ model, dtype, backend });
      }
    }
  }
  return cells;
}

function runVariantIsolated(variantArgs) {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      ['--expose-gc', variantScript, ...variantArgs],
      { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let stderr = '';
    child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
    child.on('close', async (code, signal) => {
      const resultFile = variantArgs[variantArgs.indexOf('--result-file') + 1];
      try {
        const raw = await fs.readFile(resultFile, 'utf8');
        const result = JSON.parse(raw);
        if ((code !== 0 || signal) && result.status === 'ok') {
          // keep ok
        } else if (signal === 'SIGKILL' || code === 137) {
          result.status = result.status === 'ok' ? result.status : 'error';
          result.error = result.error ?? 'Process killed (likely OOM)';
          result.error_kind = 'oom';
        }
        resolve({ result, stderr });
      } catch (error) {
        resolve({
          result: {
            status: 'error',
            error:
              signal === 'SIGKILL' || code === 137
                ? 'Process killed (likely OOM)'
                : error instanceof Error ? error.message : String(error),
            stderr_tail: stderr.slice(-500),
          },
          stderr,
        });
      }
    });
  });
}

async function benchmarkCell({ model, dtype, backend }, args) {
  const tmpDir = path.join(root, 'results', '.tmp');
  await fs.mkdir(tmpDir, { recursive: true });
  const resultFile = path.join(tmpDir, `gemma4-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);

  const variantArgs = [
    '--model-id', model.id,
    '--model-name', model.name,
    '--model-slug', model.slug,
    '--dtype', dtype,
    '--backend', backend,
    '--max-new-tokens', String(args.maxNewTokens),
    '--result-file', resultFile,
  ];
  if (args.maxPrompts != null) {
    variantArgs.push('--max-prompts', String(args.maxPrompts));
  }
  if (args.webgpuStrategy && backend === 'webgpu') {
    variantArgs.push('--webgpu-strategy', args.webgpuStrategy);
  }

  const { result } = await runVariantIsolated(variantArgs);
  try {
    await fs.unlink(resultFile);
  } catch {
    // ignore
  }
  return {
    model_slug: model.slug,
    model_name: model.name,
    dtype,
    backend_requested: backend,
    ...result,
  };
}

function printSummary(run) {
  console.log('\n' + '='.repeat(120));
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

  for (const r of run.results) {
    console.log(
      [
        (r.model_slug ?? r.model_name ?? '').slice(0, 17).padEnd(18),
        String(r.dtype).padEnd(8),
        String(r.backend_used ?? r.backend_requested ?? '-').padEnd(10),
        String(r.status).padEnd(12),
        (r.load_time_ms != null ? `${Math.round(r.load_time_ms)}ms` : '-').padStart(8),
        (r.inference?.mean_ms ?? '-').toString().padStart(10),
        (r.tokens_per_sec ?? '-').toString().padStart(8),
        (r.memory?.peak_rss_mb ?? '-').toString().padStart(8),
      ].join(' '),
    );
  }

  console.log('-'.repeat(120));
  console.log(`Wall: ${formatDuration(run.wall_time_ms)} | ok: ${run.summary.ok}/${run.summary.total}`);
  console.log('='.repeat(120));
}

async function main() {
  const args = parseArgs(process.argv);
  const cells = buildCells(args);
  console.log(`Gemma 4 benchmark — ${cells.length} variant(s), ${args.maxPrompts ?? 'all'} prompt(s), ${args.maxNewTokens} new tokens\n`);

  const wallStart = performance.now();
  const results = [];

  for (const cell of cells) {
    const label = `${cell.model.slug} ${cell.dtype} ${cell.backend}`;
    process.stdout.write(`→ ${label.padEnd(28)} … `);
    const result = await benchmarkCell(cell, args);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`ok  ${result.inference?.mean_ms}ms/prompt  ${result.tokens_per_sec} tok/s`);
    } else if (result.status === 'infer_error') {
      console.log(`infer fail (${result.error_kind ?? 'error'})`);
    } else {
      console.log(`fail: ${(result.error ?? '').slice(0, 60)}`);
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
    wall_time_ms: Math.round(performance.now() - wallStart),
    results,
    summary: {
      total: results.length,
      ok: results.filter((r) => r.status === 'ok').length,
      infer_error: results.filter((r) => r.status === 'infer_error').length,
      error: results.filter((r) => r.status === 'error').length,
    },
  };

  const outPath = args.output ?? path.join(root, 'results', `benchmark-gemma4-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(run, null, 2));
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  GEMMA4_BACKENDS,
  GEMMA4_MODELS,
  findGemma4Model,
} from '../config/gemma4-models.mjs';
import {
  RESULTS_DIR,
  SCRIPTS_DIR,
  parseCsv,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';

const workerScript = path.join(SCRIPTS_DIR, 'probe-gemma4-worker.mjs');

function parseCli() {
  const { values } = parseArgs({
    options: {
      quick: { type: 'boolean' },
      model: { type: 'string' },
      dtype: { type: 'string' },
      backend: { type: 'string' },
      'skip-fp32': { type: 'boolean' },
      hard: { type: 'boolean' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;
  const backends = parseCsv(values.backend) ?? GEMMA4_BACKENDS;
  const unknownBackends = backends.filter((backend) => !GEMMA4_BACKENDS.includes(backend));
  if (unknownBackends.length) {
    throw new Error(`Unknown backend(s): ${unknownBackends.join(', ')}`);
  }
  return {
    quick: values.quick ?? false,
    models: parseCsv(values.model),
    dtypes: parseCsv(values.dtype),
    backends,
    skipFp32: values['skip-fp32'] ?? false,
    hard: values.hard ?? false,
  };
}

function printHelp() {
  console.log(`Usage: node scripts/probe-gemma4-matrix.mjs [options]

Options:
  --quick             E2B-it q4 on all backends
  --model slug[,slug] Filter models
  --dtype d[,d]       Filter quants
  --backend b[,b]     Filter backends
  --skip-fp32         Exclude fp32
  --hard              Try extra WebGPU strategies
  -h, --help          Show this help
`);
}

function buildMatrix(args) {
  const requestedModels = args.models ?? (args.quick ? ['E2B-it'] : null);
  const models = requestedModels
    ? requestedModels.map((slug) => {
        const model = findGemma4Model(slug);
        if (!model) throw new Error(`Unknown model: ${slug}`);
        return model;
      })
    : GEMMA4_MODELS;
  const cells = [];
  for (const model of models) {
    let dtypes = (args.dtypes ?? (args.quick ? ['q4'] : model.quants))
      .filter((dtype) => model.quants.includes(dtype));
    if (args.skipFp32) dtypes = dtypes.filter((dtype) => dtype !== 'fp32');
    for (const dtype of dtypes) {
      for (const backend of args.backends) {
        cells.push({ model, dtype, backend });
      }
    }
  }
  if (!cells.length) throw new Error('No compatible probe cells selected');
  return cells;
}

async function runWorker(modelId, backend, dtype, hard) {
  const args = [modelId, backend, dtype];
  if (hard && backend === 'webgpu') args.push('--hard');
  return (await runJsonWorker(workerScript, args, {
    resultFile: false,
    failureStatus: 'load_error',
  })).result;
}

function summarize(results) {
  const summary = { ok: 0, infer_error: 0, load_error: 0, by_backend: {}, by_dtype: {} };
  for (const result of results) {
    summary[result.status] = (summary[result.status] ?? 0) + 1;
    summary.by_backend[result.backend] ??= {};
    summary.by_dtype[result.dtype] ??= {};
    summary.by_backend[result.backend][result.status] =
      (summary.by_backend[result.backend][result.status] ?? 0) + 1;
    summary.by_dtype[result.dtype][result.status] =
      (summary.by_dtype[result.dtype][result.status] ?? 0) + 1;
  }
  return summary;
}

function cell(result) {
  if (!result) return '-'.padEnd(11);
  if (result.status === 'ok') return 'ok'.padEnd(11);
  if (result.status === 'infer_error') return 'infer!'.padEnd(11);
  return 'fail'.padEnd(11);
}

function printSummary(report) {
  const models = [...new Set(report.results.map((result) => result.model_slug))];
  console.log('\n--- Matrix ---');
  for (const slug of models) {
    console.log(`\n${slug}`);
    console.log(`Quant     ${GEMMA4_BACKENDS.map((backend) => backend.padEnd(11)).join('')}`);
    const dtypes = [...new Set(
      report.results.filter((result) => result.model_slug === slug)
        .map((result) => result.dtype),
    )];
    for (const dtype of dtypes) {
      const cells = GEMMA4_BACKENDS.map((backend) =>
        cell(report.results.find((result) =>
          result.model_slug === slug
          && result.dtype === dtype
          && result.backend === backend)));
      console.log(`${dtype.padEnd(8)}  ${cells.join('  ')}`);
    }
  }
  console.log(
    `\nTotals: ${report.summary.ok} ok, `
    + `${report.summary.infer_error} infer errors, ${report.summary.load_error} load errors`,
  );
}

async function main() {
  const args = parseCli();
  if (!args) {
    printHelp();
    return;
  }
  const cells = buildMatrix(args);
  console.log(`Gemma 4 probe matrix - ${cells.length} cell(s)\n`);
  const results = [];
  for (const { model, dtype, backend } of cells) {
    const label = `${model.slug.padEnd(14)} ${dtype.padEnd(6)} ${backend.padEnd(9)}`;
    process.stdout.write(`${label} ... `);
    const result = await runWorker(model.id, backend, dtype, args.hard);
    results.push({ ...result, model_slug: model.slug, model_name: model.name });
    if (result.status === 'ok') {
      const text = (result.generated_text ?? '').replace(/\s+/g, ' ').slice(0, 48);
      console.log(`ok load=${result.load_ms}ms infer=${result.infer_ms}ms "${text}"`);
    } else {
      console.log(`${result.status}: ${(result.error ?? '').slice(0, 70)}`);
    }
  }

  const report = {
    tested_at: new Date().toISOString(),
    cells: cells.length,
    results,
    summary: summarize(results),
  };
  const outPath = path.join(RESULTS_DIR, `probe-gemma4-matrix-${Date.now()}.json`);
  await writeJson(outPath, report);
  printSummary(report);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

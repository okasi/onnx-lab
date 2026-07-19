#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import { findGemma4Model } from '../config/gemma4-models.mjs';
import {
  RESULTS_DIR,
  ROOT_DIR,
  SCRIPTS_DIR,
  parseCsv,
  positiveInteger,
  projectPath,
  readJson,
  round,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';
import {
  loadMultimodalTasks,
  summarizeMultimodalTasks,
} from '../lib/gemma4-multimodal-suite.mjs';

const workerScript = path.join(SCRIPTS_DIR, 'eval-gemma4-multimodal-worker.mjs');
const DEFAULT_MODELS = ['E2B-it', 'E4B-it'];
const DEFAULT_DTYPES = ['q4', 'q4f16'];

function parseCli() {
  const { values } = parseArgs({
    options: {
      model: { type: 'string' },
      dtype: { type: 'string' },
      modality: { type: 'string' },
      'max-tasks': { type: 'string' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;
  if (values.modality && !['image', 'audio'].includes(values.modality)) {
    throw new Error('--modality must be image or audio');
  }
  return {
    models: parseCsv(values.model) ?? DEFAULT_MODELS,
    dtypes: parseCsv(values.dtype) ?? DEFAULT_DTYPES,
    modality: values.modality ?? null,
    maxTasks: positiveInteger(values['max-tasks'], '--max-tasks'),
    output: values.output ? path.resolve(ROOT_DIR, values.output) : null,
  };
}

function printHelp() {
  console.log(`Usage: node scripts/eval-gemma4-multimodal.mjs [options]

Options:
  --model slug[,slug]     E2B-it,E4B-it (default)
  --dtype d[,d]           q4,q4f16 (default)
  --modality image|audio  Run one modality only
  --max-tasks N           Limit tasks after modality filtering
  --output path           JSON output path
  -h, --help              Show this help
`);
}

async function runWorker(model, dtype, args) {
  const workerArgs = [
    '--model-slug', model.slug,
    '--dtype', dtype,
  ];
  if (args.modality) workerArgs.push('--modality', args.modality);
  if (args.maxTasks) workerArgs.push('--max-tasks', String(args.maxTasks));
  return (await runJsonWorker(workerScript, workerArgs, { resultFile: true })).result;
}

function printSummary(run) {
  console.log(`\n${'='.repeat(120)}`);
  console.log(`GEMMA 4 MULTIMODAL EVAL (${run.modality})`);
  console.log('='.repeat(120));
  console.log(
    [
      'Model'.padEnd(10),
      'Quant'.padEnd(8),
      'Status'.padEnd(12),
      'Load'.padStart(8),
      'Pass'.padStart(8),
      'Avg ms'.padStart(10),
      'RSS MB'.padStart(8),
    ].join(' '),
  );
  console.log('-'.repeat(120));
  for (const result of run.results) {
    const inference = (result.tasks ?? [])
      .filter((task) => task.infer_ms != null)
      .map((task) => task.infer_ms);
    const average = inference.length
      ? round(inference.reduce((sum, value) => sum + value, 0) / inference.length)
      : null;
    console.log(
      [
        String(result.model_slug).padEnd(10),
        String(result.dtype).padEnd(8),
        String(result.status).padEnd(12),
        (result.load_time_ms != null ? `${Math.round(result.load_time_ms)}ms` : '-').padStart(8),
        `${result.summary?.pass ?? 0}/${result.summary?.total ?? 0}`.padStart(8),
        (average != null ? `${average}ms` : '-').padStart(10),
        String(result.memory?.peak_rss_mb ?? '-').padStart(8),
      ].join(' '),
    );
  }
  console.log('-'.repeat(120));
}

async function main() {
  const args = parseCli();
  if (!args) {
    printHelp();
    return;
  }
  const models = args.models.map((slug) => {
    const model = findGemma4Model(slug);
    if (!model) throw new Error(`Unknown model: ${slug}`);
    return model;
  });
  const cells = models.flatMap((model) =>
    args.dtypes.map((dtype) => {
      if (!model.quants.includes(dtype)) {
        throw new Error(`${model.slug} does not provide dtype ${dtype}`);
      }
      return { model, dtype };
    }));
  const suite = await readJson(projectPath('data', 'gemma4-multimodal-suite.json'));
  const tasks = loadMultimodalTasks(suite, args.modality)
    .slice(0, args.maxTasks ?? Number.POSITIVE_INFINITY);
  const taskInfo = summarizeMultimodalTasks(tasks);
  if (!taskInfo.total) {
    throw new Error('No multimodal tasks selected');
  }

  console.log(
    `Gemma 4 multimodal eval - ${cells.length} variant(s), `
    + `${taskInfo.total} task(s) [${args.modality ?? 'image+audio'}]\n`,
  );
  const startedAt = performance.now();
  const results = [];
  for (const { model, dtype } of cells) {
    process.stdout.write(`${`${model.slug} ${dtype}`.padEnd(24)} ... `);
    const result = await runWorker(model, dtype, args);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`ok pass ${result.summary.pass}/${result.summary.total}`);
    } else {
      console.log(`${result.status}: ${(result.error ?? '').slice(0, 70)}`);
    }
  }

  const run = {
    eval: 'gemma4-multimodal',
    modality: args.modality ?? 'all',
    task_count: taskInfo.total,
    max_tasks: args.maxTasks,
    tested_at: new Date().toISOString(),
    wall_time_ms: Math.round(performance.now() - startedAt),
    results,
    summary: {
      variants: results.length,
      ok: results.filter((result) => result.status === 'ok').length,
      pass_all: results.filter((result) =>
        result.summary?.total > 0 && result.summary.pass === result.summary.total).length,
    },
  };
  const suffix = args.modality ?? 'full';
  const outPath = args.output
    ?? path.join(RESULTS_DIR, `eval-gemma4-multimodal-${suffix}-${Date.now()}.json`);
  await writeJson(outPath, run);
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

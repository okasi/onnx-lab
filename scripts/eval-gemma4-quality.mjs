#!/usr/bin/env node
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import { findGemma4Model } from '../config/gemma4-models.mjs';
import {
  RESULTS_DIR,
  ROOT_DIR,
  SCRIPTS_DIR,
  parseCsv,
  positiveInteger,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';

const workerScript = path.join(SCRIPTS_DIR, 'eval-gemma4-quality-worker.mjs');
const DEFAULT_MODELS = ['E2B-it', 'E4B-it'];
const DEFAULT_DTYPES = ['q4', 'q4f16'];

function crossProduct(models, dtypes) {
  return models.flatMap((slug) => dtypes.map((dtype) => ({ slug, dtype })));
}

function parseCli() {
  const { values } = parseArgs({
    options: {
      variant: { type: 'string', multiple: true },
      model: { type: 'string' },
      dtype: { type: 'string' },
      backend: { type: 'string' },
      category: { type: 'string' },
      'max-tasks': { type: 'string' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;

  let variants;
  if (values.variant?.length) {
    variants = values.variant.map((value) => {
      const [slug, dtype, extra] = value.split(':');
      if (!slug || !dtype || extra) {
        throw new Error(`Invalid --variant "${value}"; expected model:dtype`);
      }
      return { slug, dtype };
    });
  } else {
    variants = crossProduct(
      parseCsv(values.model) ?? DEFAULT_MODELS,
      parseCsv(values.dtype) ?? DEFAULT_DTYPES,
    );
  }

  return {
    variants,
    backend: values.backend ?? 'cpu',
    category: values.category ?? null,
    maxTasks: positiveInteger(values['max-tasks'], '--max-tasks'),
    output: values.output ? path.resolve(ROOT_DIR, values.output) : null,
  };
}

function printHelp() {
  console.log(`Usage: node scripts/eval-gemma4-quality.mjs [options]

Options:
  --variant model:dtype   Run one variant; may be repeated
  --model slug[,slug]     Models (default: E2B-it,E4B-it)
  --dtype d[,d]           Quants (default: q4,q4f16)
  --backend cpu           Backend (default: cpu)
  --category NAME         Run one category only
  --max-tasks N           Limit tasks per selected category
  --output path           Results JSON path
  -h, --help              Show this help
`);
}

async function runWorker(model, dtype, args) {
  const workerArgs = [
    '--model-id', model.id,
    '--model-slug', model.slug,
    '--dtype', dtype,
    '--backend', args.backend,
  ];
  if (args.category) workerArgs.push('--category', args.category);
  if (args.maxTasks) workerArgs.push('--max-tasks', String(args.maxTasks));
  return (await runJsonWorker(workerScript, workerArgs, { resultFile: true })).result;
}

function printSummary(run) {
  console.log(`\n${'='.repeat(100)}`);
  console.log('GEMMA 4 QUALITY SUMMARY');
  console.log('='.repeat(100));
  console.log(
    ['Model', 'Quant', 'Overall', 'Writing', 'JSON', 'MCQ', 'Reading', 'Instruct', 'Summary', 'Classify', 'Pass%'].join('\t'),
  );
  console.log('-'.repeat(100));

  for (const result of run.results) {
    if (result.status !== 'ok') {
      console.log(`${result.model_slug}\t${result.dtype}\tFAIL\t-\t-\t-\t-\t-\t-`);
      continue;
    }
    const categories = result.categories;
    console.log(
      [
        result.model_slug,
        result.dtype,
        result.overall_score,
        categories.domain_writing?.summary?.mean_score ?? '-',
        categories.json_extraction?.summary?.mean_score ?? '-',
        categories.mcq?.summary?.mean_score ?? '-',
        categories.reading_comprehension?.summary?.mean_score ?? '-',
        categories.instruction_following?.summary?.mean_score ?? '-',
        categories.summarization?.summary?.mean_score ?? '-',
        categories.classification?.summary?.mean_score ?? '-',
        result.summary?.pass_rate ?? '-',
      ].join('\t'),
    );
  }
  console.log('='.repeat(100));
}

async function writeMarkdownReport(run, outPath) {
  const lines = [
    '# Gemma 4 Quality Evaluation',
    '',
    `Tested: ${run.tested_at}`,
    `Backend: ${run.backend}`,
    `Variants: ${run.results.length}`,
    '',
    '## Overall scores',
    '',
    '| Model | Quant | Overall | Pass rate | Load (s) |',
    '|-------|-------|---------|-----------|----------|',
  ];

  for (const result of run.results) {
    if (result.status !== 'ok') {
      lines.push(`| ${result.model_slug} | ${result.dtype} | **error** | - | - |`);
    } else {
      lines.push(
        `| ${result.model_slug} | ${result.dtype} | **${result.overall_score}** | `
        + `${(result.summary.pass_rate * 100).toFixed(0)}% | `
        + `${(result.load_time_ms / 1000).toFixed(1)} |`,
      );
    }
  }

  const labels = {
    domain_writing: 'Domain writing',
    json_extraction: 'JSON extraction',
    mcq: 'MCQ',
    reading_comprehension: 'Reading comprehension',
    instruction_following: 'Instruction following',
    summarization: 'Summarization',
    classification: 'Classification',
  };
  lines.push('', '## By category', '');
  for (const [category, label] of Object.entries(labels)) {
    lines.push(`### ${label}`, '', '| Model | Quant | Mean score | Pass rate |');
    lines.push('|-------|-------|------------|-----------|');
    for (const result of run.results) {
      const summary = result.categories?.[category]?.summary;
      if (result.status === 'ok' && summary) {
        lines.push(
          `| ${result.model_slug} | ${result.dtype} | ${summary.mean_score} | `
          + `${(summary.pass_rate * 100).toFixed(0)}% |`,
        );
      }
    }
    lines.push('');
  }
  lines.push(
    '## Notes',
    '',
    '- Scores use deterministic local heuristics; no external judge model is involved.',
    '- Generation uses the model chat template and greedy decoding.',
    '',
  );
  await fs.writeFile(outPath, lines.join('\n'), 'utf8');
}

async function main() {
  const args = parseCli();
  if (!args) {
    printHelp();
    return;
  }
  const variants = args.variants.map(({ slug, dtype }) => {
    const model = findGemma4Model(slug);
    if (!model) throw new Error(`Unknown model: ${slug}`);
    if (!model.quants.includes(dtype)) {
      throw new Error(`${model.slug} does not provide dtype ${dtype}`);
    }
    return { model, dtype };
  });

  console.log(`Gemma 4 quality eval - ${variants.length} variant(s), backend=${args.backend}\n`);
  const startedAt = performance.now();
  const results = [];
  for (const { model, dtype } of variants) {
    process.stdout.write(`${`${model.slug} ${dtype}`.padEnd(20)} ... `);
    const result = await runWorker(model, dtype, args);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`overall=${result.overall_score} pass=${(result.summary.pass_rate * 100).toFixed(0)}%`);
    } else {
      console.log(`failed: ${(result.error ?? '').slice(0, 70)}`);
    }
  }

  const run = {
    benchmark: 'gemma4-quality',
    tested_at: new Date().toISOString(),
    backend: args.backend,
    category: args.category,
    max_tasks: args.maxTasks,
    wall_time_ms: Math.round(performance.now() - startedAt),
    results,
  };
  const outPath = args.output ?? path.join(RESULTS_DIR, `eval-gemma4-quality-${Date.now()}.json`);
  await writeJson(outPath, run);
  if (!args.category && !args.maxTasks) {
    await writeMarkdownReport(run, path.join(ROOT_DIR, 'GEMMA4_QUALITY.md'));
  }
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

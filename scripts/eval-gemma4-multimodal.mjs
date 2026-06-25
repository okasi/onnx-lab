#!/usr/bin/env node
/**
 * Gemma 4 multimodal eval — image and/or audio on E2B/E4B × q4/q4f16 (CPU).
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { round } from '../lib/gemma4-helpers.mjs';
import { loadMultimodalTasks, summarizeMultimodalTasks } from '../lib/gemma4-multimodal-suite.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const workerScript = path.join(__dirname, 'eval-gemma4-multimodal-worker.mjs');

const DEFAULT_MODELS = ['E2B-it', 'E4B-it'];
const DEFAULT_DTYPES = ['q4', 'q4f16'];

function parseArgs(argv) {
  const args = {
    models: null,
    dtypes: null,
    modality: null,
    output: null,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') args.models = argv[++i].split(',');
    else if (arg === '--dtype') args.dtypes = argv[++i].split(',');
    else if (arg === '--modality') args.modality = argv[++i];
    else if (arg === '--output') args.output = argv[++i];
    else if (arg === '--help') {
      console.log(`Usage: node scripts/eval-gemma4-multimodal.mjs [options]

Options:
  --model slug[,slug]     E2B-it,E4B-it (default)
  --dtype d[,d]           q4,q4f16 (default)
  --modality image|audio  Run one modality only (default: both)
  --output path           JSON output path
`);
      process.exit(0);
    }
  }
  return args;
}


function runWorker(modelSlug, dtype, modality) {
  return new Promise((resolve) => {
    const tmpDir = path.join(root, 'results', '.tmp');
    const resultFile = path.join(tmpDir, `gemma4-mm-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);

    const workerArgs = [
      '--expose-gc', workerScript,
      '--model-slug', modelSlug,
      '--dtype', dtype,
      '--result-file', resultFile,
    ];
    if (modality) workerArgs.push('--modality', modality);

    fs.mkdir(tmpDir, { recursive: true }).then(() => {
      const child = spawn(process.execPath, workerArgs, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
      let stderr = '';
      child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
      child.on('close', async () => {
        try {
          const raw = await fs.readFile(resultFile, 'utf8');
          const result = JSON.parse(raw);
          resolve({ result, stderr });
        } catch (error) {
          resolve({
            result: {
              model_slug: modelSlug,
              dtype,
              modality: modality ?? 'all',
              status: 'error',
              error: error instanceof Error ? error.message : String(error),
              stderr_tail: stderr.slice(-500),
            },
            stderr,
          });
        } finally {
          try { await fs.unlink(resultFile); } catch { /* ignore */ }
        }
      });
    });
  });
}

function printSummary(run) {
  const modality = run.modality ?? 'all';
  console.log('\n' + '='.repeat(120));
  console.log(`GEMMA 4 MULTIMODAL EVAL (${modality})`);
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

  for (const r of run.results) {
    const inferValues = (r.tasks ?? []).filter((t) => t.infer_ms != null).map((t) => t.infer_ms);
    const avgInfer = inferValues.length
      ? round(inferValues.reduce((a, b) => a + b, 0) / inferValues.length)
      : null;
    console.log(
      [
        String(r.model_slug).padEnd(10),
        String(r.dtype).padEnd(8),
        String(r.status).padEnd(12),
        (r.load_time_ms != null ? `${Math.round(r.load_time_ms)}ms` : '-').padStart(8),
        (`${r.summary?.pass ?? 0}/${r.summary?.total ?? 0}`).padStart(8),
        (avgInfer != null ? `${avgInfer}ms` : '-').padStart(10),
        (r.memory?.peak_rss_mb ?? '-').toString().padStart(8),
      ].join(' '),
    );
  }
  console.log('-'.repeat(120));
}

async function main() {
  const args = parseArgs(process.argv);
  const modelSlugs = args.models ?? DEFAULT_MODELS;
  const dtypes = args.dtypes ?? DEFAULT_DTYPES;

  const suite = JSON.parse(
    await fs.readFile(path.join(root, 'data', 'gemma4-multimodal-suite.json'), 'utf8'),
  );
  const taskInfo = summarizeMultimodalTasks(loadMultimodalTasks(suite, args.modality));

  const cells = [];
  for (const slug of modelSlugs) {
    for (const dtype of dtypes) {
      cells.push({ slug, dtype });
    }
  }

  const modalityLabel = args.modality ?? 'image+audio';
  console.log(`Gemma 4 multimodal eval — ${cells.length} variant(s), ${taskInfo.total} task(s) [${modalityLabel}]\n`);

  const wallStart = performance.now();
  const results = [];

  for (const { slug, dtype } of cells) {
    process.stdout.write(`→ ${slug} ${dtype}`.padEnd(24) + ' … ');
    const { result } = await runWorker(slug, dtype, args.modality);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`ok  pass ${result.summary.pass}/${result.summary.total}`);
    } else {
      console.log(`${result.status}: ${(result.error ?? result.tasks?.find((t) => t.error)?.error ?? '').slice(0, 70)}`);
    }
  }

  const run = {
    eval: 'gemma4-multimodal',
    modality: args.modality ?? 'all',
    task_count: taskInfo.total,
    tested_at: new Date().toISOString(),
    wall_time_ms: Math.round(performance.now() - wallStart),
    results,
    summary: {
      variants: results.length,
      ok: results.filter((r) => r.status === 'ok').length,
      pass_all: results.filter((r) => r.summary?.pass === r.summary?.total).length,
    },
  };

  const suffix = args.modality ?? 'full';
  const outPath = args.output ?? path.join(root, 'results', `eval-gemma4-multimodal-${suffix}-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(run, null, 2));
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

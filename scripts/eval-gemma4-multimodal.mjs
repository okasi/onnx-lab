#!/usr/bin/env node
/**
 * Gemma 4 multimodal eval — image + audio on E2B/E4B × q4/q4f16 (CPU).
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { GEMMA4_MODELS } from '../config/gemma4-models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const workerScript = path.join(__dirname, 'eval-gemma4-multimodal-worker.mjs');

const DEFAULT_MODELS = ['E2B-it', 'E4B-it'];
const DEFAULT_DTYPES = ['q4', 'q4f16'];

function parseArgs(argv) {
  const args = {
    models: null,
    dtypes: null,
    output: null,
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') args.models = argv[++i].split(',');
    else if (arg === '--dtype') args.dtypes = argv[++i].split(',');
    else if (arg === '--output') args.output = argv[++i];
    else if (arg === '--help') {
      console.log(`Usage: node scripts/eval-gemma4-multimodal.mjs [options]

Options:
  --model slug[,slug]   E2B-it,E4B-it (default)
  --dtype d[,d]         q4,q4f16 (default)
  --output path         JSON output path
`);
      process.exit(0);
    }
  }
  return args;
}

function runWorker(modelSlug, dtype) {
  return new Promise((resolve) => {
    const tmpDir = path.join(root, 'results', '.tmp');
    const resultFile = path.join(tmpDir, `gemma4-mm-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);

    fs.mkdir(tmpDir, { recursive: true }).then(() => {
      const child = spawn(
        process.execPath,
        ['--expose-gc', workerScript, '--model-slug', modelSlug, '--dtype', dtype, '--result-file', resultFile],
        { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] },
      );
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
  console.log('\n' + '='.repeat(110));
  console.log('GEMMA 4 MULTIMODAL EVAL');
  console.log('='.repeat(110));
  console.log(
    [
      'Model'.padEnd(10),
      'Quant'.padEnd(8),
      'Status'.padEnd(12),
      'Load'.padStart(8),
      'Image'.padStart(8),
      'Audio'.padStart(8),
      'Pass'.padStart(6),
      'RSS MB'.padStart(8),
    ].join(' '),
  );
  console.log('-'.repeat(110));

  for (const r of run.results) {
    const image = r.tasks?.find((t) => t.modality === 'image');
    const audio = r.tasks?.find((t) => t.modality === 'audio');
    console.log(
      [
        String(r.model_slug).padEnd(10),
        String(r.dtype).padEnd(8),
        String(r.status).padEnd(12),
        (r.load_time_ms != null ? `${Math.round(r.load_time_ms)}ms` : '-').padStart(8),
        (image?.infer_ms != null ? `${Math.round(image.infer_ms)}ms` : image?.status ?? '-').padStart(8),
        (audio?.infer_ms != null ? `${Math.round(audio.infer_ms)}ms` : audio?.status ?? '-').padStart(8),
        (`${r.summary?.pass ?? 0}/${r.summary?.total ?? 0}`).padStart(6),
        (r.memory?.peak_rss_mb ?? '-').toString().padStart(8),
      ].join(' '),
    );
  }
  console.log('-'.repeat(110));
}

async function main() {
  const args = parseArgs(process.argv);
  const modelSlugs = args.models ?? DEFAULT_MODELS;
  const dtypes = args.dtypes ?? DEFAULT_DTYPES;

  const cells = [];
  for (const slug of modelSlugs) {
    for (const dtype of dtypes) {
      cells.push({ slug, dtype });
    }
  }

  console.log(`Gemma 4 multimodal eval — ${cells.length} variant(s), image + audio tasks\n`);

  const wallStart = performance.now();
  const results = [];

  for (const { slug, dtype } of cells) {
    process.stdout.write(`→ ${slug} ${dtype}`.padEnd(24) + ' … ');
    const { result } = await runWorker(slug, dtype);
    results.push(result);
    if (result.status === 'ok') {
      const img = result.tasks?.find((t) => t.modality === 'image');
      const aud = result.tasks?.find((t) => t.modality === 'audio');
      console.log(`ok  img ${img?.infer_ms}ms  aud ${aud?.infer_ms}ms  pass ${result.summary.pass}/${result.summary.total}`);
    } else {
      console.log(`${result.status}: ${(result.error ?? result.tasks?.find((t) => t.error)?.error ?? '').slice(0, 70)}`);
    }
  }

  const run = {
    eval: 'gemma4-multimodal',
    tested_at: new Date().toISOString(),
    wall_time_ms: Math.round(performance.now() - wallStart),
    results,
    summary: {
      total: results.length,
      ok: results.filter((r) => r.status === 'ok').length,
      pass_all: results.filter((r) => r.summary?.pass === r.summary?.total).length,
    },
  };

  const outPath = args.output ?? path.join(root, 'results', `eval-gemma4-multimodal-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(run, null, 2));
  printSummary(run);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

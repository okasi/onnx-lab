#!/usr/bin/env node
/**
 * Matrix test: Gemma 4 ONNX LLMs × quants × backends (CPU, wasm-jsep, wasm, WebGPU).
 *
 * Flags:
 *   --quick          E2B-it only, q4 quant, all backends
 *   --model <slug>   Filter to one model slug (e.g. E2B-it, E4B-qat-mobile)
 *   --dtype <list>   Comma-separated quants
 *   --backend <list> Comma-separated backends
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  GEMMA4_BACKENDS,
  GEMMA4_MODELS,
  findGemma4Model,
} from '../config/gemma4-models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const worker = path.join(__dirname, 'probe-gemma4-worker.mjs');

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  if (i === -1) return null;
  return process.argv[i + 1] ?? null;
}

function hasFlag(flag) {
  return process.argv.includes(flag);
}

function parseList(value) {
  return value?.split(',').map((s) => s.trim()).filter(Boolean) ?? null;
}

function buildMatrix() {
  if (hasFlag('--quick')) {
    const model = findGemma4Model('E2B-it');
    return GEMMA4_BACKENDS.map((backend) => ({ model, dtype: 'q4', backend }));
  }

  const modelFilter = argValue('--model');
  const dtypeFilter = parseList(argValue('--dtype'));
  const backendFilter = parseList(argValue('--backend')) ?? GEMMA4_BACKENDS;

  let models = GEMMA4_MODELS;
  if (modelFilter) {
    const m = findGemma4Model(modelFilter);
    if (!m) {
      throw new Error(`Unknown model filter: ${modelFilter}`);
    }
    models = [m];
  }

  const cells = [];
  for (const model of models) {
    const quants = dtypeFilter ?? model.quants;
    for (const dtype of quants) {
      for (const backend of backendFilter) {
        cells.push({ model, dtype, backend });
      }
    }
  }
  return cells;
}

async function main() {
  const cells = buildMatrix();
  console.log(`Gemma 4 probe matrix — ${cells.length} cell(s)\n`);

  const results = [];
  for (const { model, dtype, backend } of cells) {
    const label = `${model.slug.padEnd(14)} ${dtype.padEnd(6)} ${backend.padEnd(9)}`;
    process.stdout.write(`→ ${label} … `);
    const r = await runWorker(model.id, backend, dtype);
    results.push({ ...r, model_slug: model.slug, model_name: model.name });
    if (r.status === 'ok') {
      const text = (r.generated_text ?? '').replace(/\s+/g, ' ').slice(0, 48);
      console.log(`ok  load=${r.load_ms}ms infer=${r.infer_ms}ms  "${text}"`);
    } else if (r.status === 'infer_error') {
      console.log(`infer fail (${r.error_kind ?? 'error'}): ${(r.error ?? '').slice(0, 56)}`);
    } else {
      console.log(`load fail: ${(r.error ?? '').slice(0, 70)}`);
    }
  }

  const report = {
    tested_at: new Date().toISOString(),
    cells: cells.length,
    results,
    summary: summarize(results),
  };

  const outPath = path.join(root, 'results', `probe-gemma4-matrix-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(report, null, 2));

  printSummary(report);
  console.log(`\nWrote ${outPath}`);
}

function runWorker(modelId, backend, dtype) {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      ['--expose-gc', worker, modelId, backend, dtype],
      { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] },
    );
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        resolve({
          model_id: modelId,
          backend,
          dtype,
          status: 'load_error',
          error: (stderr.trim() || stdout.trim() || 'parse error').slice(0, 400),
        });
      }
    });
  });
}

function summarize(results) {
  const byBackend = {};
  const byDtype = {};
  for (const r of results) {
    byBackend[r.backend] ??= { ok: 0, infer_error: 0, load_error: 0 };
    byDtype[r.dtype] ??= { ok: 0, infer_error: 0, load_error: 0 };
    byBackend[r.backend][r.status] = (byBackend[r.backend][r.status] ?? 0) + 1;
    byDtype[r.dtype][r.status] = (byDtype[r.dtype][r.status] ?? 0) + 1;
  }
  return {
    ok: results.filter((r) => r.status === 'ok').length,
    infer_error: results.filter((r) => r.status === 'infer_error').length,
    load_error: results.filter((r) => r.status === 'load_error').length,
    by_backend: byBackend,
    by_dtype: byDtype,
  };
}

function printSummary(report) {
  const models = [...new Set(report.results.map((r) => r.model_slug))];
  console.log('\n--- Matrix ---');
  for (const slug of models) {
    console.log(`\n${slug}`);
    console.log('Quant     ' + GEMMA4_BACKENDS.map((b) => b.padEnd(11)).join(''));
    const rows = [...new Set(report.results.filter((r) => r.model_slug === slug).map((r) => r.dtype))];
    for (const dtype of rows) {
      const cells = GEMMA4_BACKENDS.map((backend) => {
        const r = report.results.find(
          (x) => x.model_slug === slug && x.dtype === dtype && x.backend === backend,
        );
        return cell(r);
      });
      console.log(`${dtype.padEnd(8)}  ${cells.join('  ')}`);
    }
  }
  console.log(
    `\nTotals: ${report.summary.ok} ok, ${report.summary.infer_error} infer errors, ${report.summary.load_error} load errors`,
  );
}

function cell(r) {
  if (!r) return '—'.padEnd(11);
  if (r.status === 'ok') return 'ok'.padEnd(11);
  if (r.status === 'infer_error') return 'infer!'.padEnd(11);
  return 'fail'.padEnd(11);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

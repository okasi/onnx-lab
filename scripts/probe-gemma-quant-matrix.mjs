#!/usr/bin/env node
/**
 * Matrix test: all EmbeddingGemma quants on CPU + WebGPU.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const worker = path.join(__dirname, 'probe-gemma-quant-worker.mjs');

const QUANTS = ['bnb4', 'int8', 'q4', 'q4f16', 'q8', 'uint8', 'fp32', 'fp16'];
const SMOKE_DOCS = 1;
const FULL_DOCS = Number(process.argv.includes('--full')
  ? process.argv[process.argv.indexOf('--full') + 1] || 5
  : 0);

async function main() {
  const docCount = FULL_DOCS || SMOKE_DOCS;
  console.log(`EmbeddingGemma quant matrix — CPU + WebGPU (${docCount} doc(s) each)\n`);

  const results = [];
  for (const dtype of QUANTS) {
    for (const backend of ['cpu', 'webgpu']) {
      process.stdout.write(`→ ${dtype.padEnd(6)} ${backend.padEnd(7)} … `);
      const r = await runWorker(backend, dtype, docCount);
      results.push(r);
      if (r.status === 'ok') {
        console.log(`ok  load=${r.load_ms}ms infer=${r.infer_ms}ms/doc${r.quality != null ? ` q=${r.quality}` : ''}`);
      } else {
        console.log(`fail: ${(r.error ?? '').slice(0, 70)}`);
      }
    }
  }

  const bothOk = QUANTS.filter((d) => {
    const cpu = results.find((r) => r.dtype === d && r.backend === 'cpu');
    const gpu = results.find((r) => r.dtype === d && r.backend === 'webgpu');
    return cpu?.status === 'ok' && gpu?.status === 'ok';
  });

  const report = {
    tested_at: new Date().toISOString(),
    documents_per_run: docCount,
    quants: QUANTS,
    results,
    works_on_both: bothOk,
    summary: buildSummary(results, bothOk),
  };

  const outPath = path.join(root, 'results', `probe-gemma-quant-matrix-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(report, null, 2));

  printTable(report);
  console.log(`\nWrote ${outPath}`);

  if (!FULL_DOCS && bothOk.length) {
    console.log(`\nRe-running ${bothOk.length} dual-backend quants with 5 docs for quality…`);
    const fullResults = [];
    for (const dtype of bothOk) {
      for (const backend of ['cpu', 'webgpu']) {
        process.stdout.write(`→ ${dtype} ${backend} (5 docs) … `);
        const r = await runWorker(backend, dtype, 5);
        fullResults.push(r);
        console.log(r.status === 'ok' ? `ok q=${r.quality} XL-R@5 cpu/gpu n/a` : `fail`);
      }
    }
    report.full_quality_runs = fullResults;
    report.full_summary = buildDualComparison(fullResults);
    await fs.writeFile(outPath, JSON.stringify(report, null, 2));
    printDualTable(report.full_summary);
  }
}

function runWorker(backend, dtype, maxTexts) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['--expose-gc', worker, backend, dtype, String(maxTexts)], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        resolve({ backend, dtype, status: 'error', error: stderr.trim() || stdout.trim() || 'parse error' });
      }
    });
  });
}

function buildSummary(results, bothOk) {
  const cpuOnly = [];
  const webgpuOnly = [];
  const neither = [];
  for (const d of QUANTS) {
    const cpu = results.find((r) => r.dtype === d && r.backend === 'cpu')?.status === 'ok';
    const gpu = results.find((r) => r.dtype === d && r.backend === 'webgpu')?.status === 'ok';
    if (cpu && gpu) continue;
    if (cpu) cpuOnly.push(d);
    else if (gpu) webgpuOnly.push(d);
    else neither.push(d);
  }
  return { bothOk, cpuOnly, webgpuOnly, neither };
}

function buildDualComparison(fullResults) {
  const rows = [];
  const dtypes = [...new Set(fullResults.map((r) => r.dtype))];
  for (const dtype of dtypes) {
    const cpu = fullResults.find((r) => r.dtype === dtype && r.backend === 'cpu');
    const gpu = fullResults.find((r) => r.dtype === dtype && r.backend === 'webgpu');
    if (cpu?.status !== 'ok' || gpu?.status !== 'ok') continue;
    rows.push({
      dtype,
      cpu_infer_ms: cpu.infer_ms,
      webgpu_infer_ms: gpu.infer_ms,
      speed_ratio: round(gpu.infer_ms / cpu.infer_ms, 1),
      cpu_quality: cpu.quality,
      webgpu_quality: gpu.quality,
      cpu_xl_r5: cpu.xl_r5,
      webgpu_xl_r5: gpu.xl_r5,
      cpu_rss_mb: cpu.peak_rss_mb,
    });
  }
  return rows;
}

function round(n, d) {
  return Number(Number(n).toFixed(d));
}

function printTable(report) {
  console.log('\n--- Matrix ---');
  console.log('Quant     CPU          WebGPU');
  console.log('--------  -----------  -----------');
  for (const d of QUANTS) {
    const cpu = report.results.find((r) => r.dtype === d && r.backend === 'cpu');
    const gpu = report.results.find((r) => r.dtype === d && r.backend === 'webgpu');
    console.log(
      `${d.padEnd(8)}  ${cell(cpu).padEnd(11)}  ${cell(gpu)}`,
    );
  }
  console.log(`\nWorks on BOTH: ${report.works_on_both.join(', ') || 'none'}`);
}

function cell(r) {
  if (!r) return '—';
  return r.status === 'ok' ? 'ok' : 'fail';
}

function printDualTable(rows) {
  if (!rows?.length) return;
  console.log('\n--- Dual-backend quants (5 docs) ---');
  console.log('Quant   CPU ms  GPU ms  Ratio  CPU Q  GPU Q  CPU XL-R@5  GPU XL-R@5  RSS MB');
  for (const r of rows) {
    console.log(
      `${r.dtype.padEnd(6)}  ${String(r.cpu_infer_ms).padEnd(6)}  ${String(r.webgpu_infer_ms).padEnd(6)}  ${String(r.speed_ratio).padEnd(5)}  ` +
        `${r.cpu_quality}  ${r.webgpu_quality}  ${r.cpu_xl_r5}       ${r.webgpu_xl_r5}        ${r.cpu_rss_mb ?? '—'}`,
    );
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

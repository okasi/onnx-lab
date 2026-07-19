#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  RESULTS_DIR,
  SCRIPTS_DIR,
  positiveInteger,
  round,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';

const workerScript = path.join(SCRIPTS_DIR, 'probe-gemma-quant-worker.mjs');
const QUANTS = ['bnb4', 'int8', 'q4', 'q4f16', 'q8', 'uint8', 'fp32', 'fp16'];

function parseCli() {
  const { values } = parseArgs({
    options: {
      full: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;
  return {
    documentCount: positiveInteger(values.full ?? '1', '--full'),
    rerunQuality: !values.full,
  };
}

async function runWorker(backend, dtype, maxTexts) {
  return (await runJsonWorker(
    workerScript,
    [backend, dtype, String(maxTexts)],
    { resultFile: false },
  )).result;
}

function buildSummary(results, bothOk) {
  const summary = { bothOk, cpuOnly: [], webgpuOnly: [], neither: [] };
  for (const dtype of QUANTS) {
    const cpu = results.find((result) =>
      result.dtype === dtype && result.backend === 'cpu')?.status === 'ok';
    const gpu = results.find((result) =>
      result.dtype === dtype && result.backend === 'webgpu')?.status === 'ok';
    if (cpu && gpu) continue;
    if (cpu) summary.cpuOnly.push(dtype);
    else if (gpu) summary.webgpuOnly.push(dtype);
    else summary.neither.push(dtype);
  }
  return summary;
}

function buildDualComparison(results) {
  const rows = [];
  for (const dtype of [...new Set(results.map((result) => result.dtype))]) {
    const cpu = results.find((result) =>
      result.dtype === dtype && result.backend === 'cpu');
    const gpu = results.find((result) =>
      result.dtype === dtype && result.backend === 'webgpu');
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

function printTable(report) {
  console.log('\n--- Matrix ---');
  console.log('Quant     CPU          WebGPU');
  console.log('--------  -----------  -----------');
  for (const dtype of QUANTS) {
    const cpu = report.results.find((result) =>
      result.dtype === dtype && result.backend === 'cpu');
    const gpu = report.results.find((result) =>
      result.dtype === dtype && result.backend === 'webgpu');
    const cell = (result) => result?.status === 'ok' ? 'ok' : 'fail';
    console.log(`${dtype.padEnd(8)}  ${cell(cpu).padEnd(11)}  ${cell(gpu)}`);
  }
  console.log(`\nWorks on both: ${report.works_on_both.join(', ') || 'none'}`);
}

function printDualTable(rows) {
  if (!rows.length) return;
  console.log('\n--- Dual-backend quants (5 docs) ---');
  console.log('Quant   CPU ms  GPU ms  Ratio  CPU Q  GPU Q  CPU XL-R@5  GPU XL-R@5  RSS MB');
  for (const row of rows) {
    console.log(
      `${row.dtype.padEnd(6)}  ${String(row.cpu_infer_ms).padEnd(6)}  `
      + `${String(row.webgpu_infer_ms).padEnd(6)}  ${String(row.speed_ratio).padEnd(5)}  `
      + `${row.cpu_quality}  ${row.webgpu_quality}  ${row.cpu_xl_r5}       `
      + `${row.webgpu_xl_r5}        ${row.cpu_rss_mb ?? '-'}`,
    );
  }
}

async function main() {
  const args = parseCli();
  if (!args) {
    console.log('Usage: node scripts/probe-gemma-quant-matrix.mjs [--full N]');
    return;
  }
  console.log(
    `EmbeddingGemma quant matrix - CPU + WebGPU (${args.documentCount} doc(s) each)\n`,
  );
  const results = [];
  for (const dtype of QUANTS) {
    for (const backend of ['cpu', 'webgpu']) {
      process.stdout.write(`${dtype.padEnd(6)} ${backend.padEnd(7)} ... `);
      const result = await runWorker(backend, dtype, args.documentCount);
      results.push(result);
      console.log(
        result.status === 'ok'
          ? `ok load=${result.load_ms}ms infer=${result.infer_ms}ms/doc`
          : `failed: ${(result.error ?? '').slice(0, 70)}`,
      );
    }
  }
  const bothOk = QUANTS.filter((dtype) =>
    ['cpu', 'webgpu'].every((backend) =>
      results.find((result) =>
        result.dtype === dtype && result.backend === backend)?.status === 'ok'));
  const report = {
    tested_at: new Date().toISOString(),
    documents_per_run: args.documentCount,
    quants: QUANTS,
    results,
    works_on_both: bothOk,
    summary: buildSummary(results, bothOk),
  };
  const outPath = path.join(RESULTS_DIR, `probe-gemma-quant-matrix-${Date.now()}.json`);
  printTable(report);

  if (args.rerunQuality && bothOk.length) {
    console.log(`\nRe-running ${bothOk.length} dual-backend quants with 5 docs...`);
    report.full_quality_runs = [];
    for (const dtype of bothOk) {
      for (const backend of ['cpu', 'webgpu']) {
        process.stdout.write(`${dtype} ${backend} ... `);
        const result = await runWorker(backend, dtype, 5);
        report.full_quality_runs.push(result);
        console.log(result.status === 'ok' ? `ok quality=${result.quality}` : 'failed');
      }
    }
    report.full_summary = buildDualComparison(report.full_quality_runs);
    printDualTable(report.full_summary);
  }
  await writeJson(outPath, report);
  console.log(`\nWrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

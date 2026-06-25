#!/usr/bin/env node
/**
 * Compare EmbeddingGemma q4: CPU (Node) vs WebGPU (Chrome).
 */
import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const worker = path.join(__dirname, 'compare-webgpu-q4-worker.mjs');
const CPU_BASELINE = path.join(root, 'results/benchmark-1781714103604.json');

async function main() {
  const maxTexts = process.argv.includes('--max-texts')
    ? Number(process.argv[process.argv.indexOf('--max-texts') + 1])
    : 54;

  console.log(`Comparing EmbeddingGemma q4: CPU vs WebGPU (${maxTexts} docs)\n`);

  const cpu = await runNodeBench('cpu', maxTexts);
  const webgpu = await runBrowserBench(maxTexts);

  const out = {
    model: 'onnx-community/embeddinggemma-300m-ONNX',
    dtype: 'q4',
    documents: maxTexts,
    cpu,
    webgpu,
    comparison: compare(cpu, webgpu),
    cpu_full_corpus_baseline: await loadCpuBaseline(),
  };

  const outPath = path.join(root, 'results', `compare-q4-cpu-webgpu-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(out, null, 2));
  printReport(out);
  console.log(`\nWrote ${outPath}`);
}

async function loadCpuBaseline() {
  try {
    const data = JSON.parse(await fs.readFile(CPU_BASELINE, 'utf8'));
    const row = data.results.find(
      (r) => r.model_id?.includes('embeddinggemma') && r.dtype === 'q4' && r.backend_used === 'cpu',
    );
    if (!row) return null;
    return {
      source: 'benchmark-1781714103604.json',
      documents: 54,
      load_ms: row.load_time_ms,
      total_ms: row.total_time_ms,
      mean_ms: row.inference?.mean_ms,
      peak_rss_mb: row.memory?.peak_rss_mb,
      quality: row.quality?.composite_score,
      xling: row.quality?.cross_lingual_pairs?.mean_cosine,
      xl_r5: row.quality?.cross_lingual_recall_at_5,
      r5: row.quality?.recall_at_5,
    };
  } catch {
    return null;
  }
}

function runNodeBench(backend, maxTexts) {
  return runWorker([backend, String(maxTexts)]);
}

function runBrowserBench(maxTexts) {
  return runWorker(['webgpu', String(maxTexts)]);
}

function runWorker(args) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--expose-gc', worker, ...args], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', (code) => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        reject(new Error(stderr || stdout || `exit ${code}`));
      }
    });
  });
}

function compare(cpu, webgpu) {
  if (cpu.status !== 'ok' || webgpu.status !== 'ok') {
    return { note: 'One or both runs failed', cpu: cpu.status, webgpu: webgpu.status };
  }
  const speedRatio = webgpu.inference.mean_ms / cpu.inference.mean_ms;
  const qualityDelta = webgpu.quality.composite_score - cpu.quality.composite_score;
  const cosineMaxDiff = webgpu.max_cosine_diff_vs_cpu ?? null;
  return {
    speed_ratio_webgpu_over_cpu: round(speedRatio, 2),
    webgpu_slower_by: `${round((speedRatio - 1) * 100, 0)}%`,
    quality_delta_composite: round(qualityDelta, 4),
    embeddings_match_cpu: cosineMaxDiff !== null ? cosineMaxDiff < 0.001 : null,
    max_cosine_diff_vs_cpu: cosineMaxDiff,
    memory_note: 'CPU peak RSS is Node process; WebGPU memory is Chrome GPU process (not directly comparable)',
  };
}

function round(n, d = 2) {
  return Number(n.toFixed(d));
}

function printReport(out) {
  const { cpu, webgpu, comparison } = out;
  console.log('--- Speed ---');
  console.log(formatRow('Load', cpu.load_time_ms, webgpu.load_time_ms, 'ms'));
  console.log(formatRow('Mean / doc', cpu.inference?.mean_ms, webgpu.inference?.mean_ms, 'ms'));
  console.log(formatRow('Total', cpu.total_time_ms, webgpu.total_time_ms, 'ms'));
  console.log('\n--- Memory ---');
  console.log(`CPU peak RSS:     ${cpu.memory?.peak_rss_mb ?? '—'} MB`);
  console.log(`WebGPU Chrome RSS: ${webgpu.chrome_rss_mb ?? '—'} MB (approx, browser process)`);
  console.log('\n--- Quality (same corpus) ---');
  if (cpu.quality && webgpu.quality) {
    console.log(formatRow('Composite', cpu.quality.composite_score, webgpu.quality.composite_score));
    console.log(formatRow('XLing', cpu.quality.cross_lingual_pairs?.mean_cosine, webgpu.quality.cross_lingual_pairs?.mean_cosine));
    console.log(formatRow('XL-R@5', cpu.quality.cross_lingual_recall_at_5, webgpu.quality.cross_lingual_recall_at_5));
    console.log(formatRow('R@5', cpu.quality.recall_at_5, webgpu.quality.recall_at_5));
  }
  if (comparison.max_cosine_diff_vs_cpu != null) {
    console.log(`\nMax per-doc cosine diff vs CPU: ${comparison.max_cosine_diff_vs_cpu}`);
  }
  console.log('\n--- Verdict ---');
  console.log(JSON.stringify(comparison, null, 2));
}

function formatRow(label, cpuVal, webVal, unit = '') {
  const u = unit ? ` ${unit}` : '';
  return `${label.padEnd(14)} CPU ${cpuVal ?? '—'}${u}  |  WebGPU ${webVal ?? '—'}${u}`;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

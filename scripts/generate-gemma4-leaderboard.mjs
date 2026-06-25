#!/usr/bin/env node
/**
 * Generate GEMMA4_LEADERBOARD.md from all results/benchmark-gemma4-*.json files.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function variantKey(r) {
  return `${r.model_id ?? r.model_slug}:${r.dtype}:${r.backend_requested ?? r.backend_used}`;
}

async function main() {
  const resultsDir = path.join(root, 'results');
  const files = (await fs.readdir(resultsDir))
    .filter((f) => f.startsWith('benchmark-gemma4-') && f.endsWith('.json'))
    .sort();

  if (!files.length) {
    console.error('No benchmark-gemma4-*.json files found');
    process.exit(1);
  }

  const merged = new Map();
  const runs = [];
  for (const file of files) {
    const run = JSON.parse(await fs.readFile(path.join(resultsDir, file), 'utf8'));
    runs.push({ file, tested_at: run.tested_at, count: run.results?.length ?? 0 });
    for (const r of run.results ?? []) {
      if (!r.model_id && !r.model_slug) continue;
      if (!r.dtype || !r.backend_requested) continue;
      merged.set(variantKey(r), { ...r, source_file: file });
    }
  }

  const rows = [...merged.values()].sort((a, b) => {
    const ma = a.model_slug ?? a.model_id ?? '';
    const mb = b.model_slug ?? b.model_id ?? '';
    return ma.localeCompare(mb) || a.dtype.localeCompare(b.dtype) || String(a.backend_requested).localeCompare(String(b.backend_requested));
  });

  const ok = rows.filter((r) => r.status === 'ok');
  const bySpeed = [...ok].sort((a, b) => (a.inference?.mean_ms ?? 1e9) - (b.inference?.mean_ms ?? 1e9));

  let md = `# Gemma 4 ONNX LLM Leaderboard\n\n`;
  md += `Merged from ${files.length} benchmark run(s) (${rows.length} unique variants).\n\n`;

  md += `## Results\n\n`;
  md += `| Model | Quant | Backend | Status | Load (ms) | ms/prompt | tok/s | RSS (MB) | Notes |\n`;
  md += `|-------|-------|---------|--------|-----------|-----------|-------|----------|-------|\n`;

  for (const r of rows) {
    const notes = [];
    if (r.webgpu_dtype_fallback) notes.push(`f16→${r.webgpu_dtype_fallback.to}`);
    if (r.webgpu_strategy) notes.push(r.webgpu_strategy);
    if (r.error_kind) notes.push(r.error_kind);
    if (r.load_status === 'ok' && r.status !== 'ok') notes.push('load-only');
    md += `| ${r.model_slug ?? '-'} | ${r.dtype} | ${r.backend_used ?? r.backend_requested} | ${r.status} | `;
    md += `${r.load_time_ms ?? '-'} | ${r.inference?.mean_ms ?? '-'} | ${r.tokens_per_sec ?? '-'} | `;
    md += `${r.memory?.peak_rss_mb ?? '-'} | ${notes.join('; ') || '-'} |\n`;
  }

  if (bySpeed.length) {
    md += `\n## Fastest CPU / working backends (mean ms/prompt)\n\n`;
    for (const r of bySpeed.slice(0, 8)) {
      md += `- **${r.model_slug} ${r.dtype} ${r.backend_used}** — ${r.inference.mean_ms} ms/prompt, ${r.tokens_per_sec} tok/s, RSS ${r.memory?.peak_rss_mb ?? '?'} MB\n`;
    }
  }

  md += `\n## Backend compatibility (this environment)\n\n`;
  md += `| Backend | E2B q4 | E2B q4f16 | E4B q4 | Mobile q2f16 |\n`;
  md += `|---------|--------|-----------|--------|---------------|\n`;
  const cell = (slug, dtype, backend) => {
    const r = rows.find((x) => x.model_slug === slug && x.dtype === dtype && (x.backend_requested ?? x.backend_used) === backend);
    if (!r) return '—';
    if (r.status === 'ok') return 'ok';
    if (r.status === 'infer_error') return 'load only';
    return r.error_kind ?? 'fail';
  };
  md += `| **cpu** | ${cell('E2B-it', 'q4', 'cpu')} | ${cell('E2B-it', 'q4f16', 'cpu')} | ${cell('E4B-it', 'q4', 'cpu')} | ${cell('E2B-qat-mobile', 'q2f16', 'cpu')} |\n`;
  md += `| **wasm-jsep** | ${cell('E2B-it', 'q4', 'wasm-jsep')} | ${cell('E2B-it', 'q4f16', 'wasm-jsep')} | — | — |\n`;
  md += `| **wasm** | ${cell('E2B-it', 'q4', 'wasm')} | — | — | — |\n`;
  md += `| **webgpu** | — | — | — | — |\n`;

  md += `\n## Failed / partial variants\n\n`;
  const failed = rows.filter((r) => r.status !== 'ok');
  if (!failed.length) {
    md += `_None._\n`;
  } else {
    for (const r of failed) {
      md += `- ${r.model_slug ?? r.model_id} ${r.dtype} ${r.backend_requested}: **${r.status}** (${r.error_kind ?? 'error'})`;
      if (r.load_time_ms) md += ` — loaded in ${Math.round(r.load_time_ms)}ms`;
      md += `\n`;
    }
  }

  md += `\n## Source runs\n\n`;
  for (const r of runs) {
    md += `- \`${r.file}\` (${r.tested_at ?? '?'}, ${r.count} variants)\n`;
  }

  const outPath = path.join(root, 'GEMMA4_LEADERBOARD.md');
  await fs.writeFile(outPath, md);
  console.log(`Wrote ${outPath} (${rows.length} variants)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

#!/usr/bin/env node
/**
 * Generate GEMMA4_LEADERBOARD.md from all results/benchmark-gemma4-*.json files.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  ROOT_DIR,
  RESULTS_DIR,
  parseCsv,
  projectPath,
  readJson,
} from '../lib/benchmark-support.mjs';

function variantKey(r) {
  return `${r.model_id ?? r.model_slug}:${r.dtype}:${r.backend_requested ?? r.backend_used}`;
}

async function main() {
  const { values } = parseArgs({
    options: {
      input: { type: 'string' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) {
    console.log('Usage: node scripts/generate-gemma4-leaderboard.mjs [--input a.json,b.json] [--output file]');
    return;
  }
  const files = values.input
    ? parseCsv(values.input).map((file) => path.resolve(ROOT_DIR, file))
    : (await fs.readdir(RESULTS_DIR))
        .filter((file) => file.startsWith('benchmark-gemma4-') && file.endsWith('.json'))
        .sort()
        .map((file) => path.join(RESULTS_DIR, file));

  if (!files.length) {
    console.error('No benchmark-gemma4-*.json files found');
    process.exit(1);
  }

  const merged = new Map();
  const runs = [];
  for (const filePath of files) {
    const run = await readJson(filePath);
    const file = path.basename(filePath);
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
    md += `\n## Fastest working variants\n\n`;
    for (const r of bySpeed.slice(0, 8)) {
      md += `- **${r.model_slug} ${r.dtype} ${r.backend_used}** — ${r.inference.mean_ms} ms/prompt, ${r.tokens_per_sec} tok/s, RSS ${r.memory?.peak_rss_mb ?? '?'} MB\n`;
    }
  }

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

  const outPath = values.output
    ? path.resolve(ROOT_DIR, values.output)
    : projectPath('GEMMA4_LEADERBOARD.md');
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, md);
  console.log(`Wrote ${outPath} (${rows.length} variants)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

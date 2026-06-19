#!/usr/bin/env node
/**
 * Generates LEADERBOARD.md from benchmark result JSON files.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { MODELS } from '../config/models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const resultsDir = path.join(root, 'results');
const outPath = path.join(root, 'LEADERBOARD.md');
const activeModelIds = new Set(MODELS.map((m) => m.id));

function filterActiveModels(results) {
  return results.filter((r) => activeModelIds.has(r.model_id));
}

function formatDuration(ms) {
  if (!ms) return '—';
  const sec = Math.floor(ms / 1000);
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function fmt(v, digits = 4) {
  if (v === null || v === undefined || v === '—') return '—';
  if (typeof v === 'number') return v.toFixed(digits);
  return String(v);
}

function shortError(err) {
  if (!err) return '—';
  const line = err.split('\n')[0];
  if (line.includes('bad_alloc')) return 'OOM (std::bad_alloc)';
  if (line.includes('GatherBlockQuantized')) return 'GatherBlockQuantized not in WASM';
  if (line.includes('onnx_data') || line.includes('MountedFiles')) return 'Missing .onnx_data shard';
  if (line.includes('float16')) return 'FP16 tensor type mismatch';
  if (line.includes('SimplifiedLayerNormFusion')) return 'FP16 graph fusion error';
  return line.slice(0, 80);
}

function rowMetrics(r) {
  const q = r.quality;
  return {
    dim: r.embedding_dim ?? '—',
    load_s: r.load_time_ms ? (r.load_time_ms / 1000).toFixed(1) : '—',
    total: formatDuration(r.total_time_ms),
    ms_doc: q ? r.inference?.mean_ms : r.inference?.mean_ms ?? '—',
    p95: r.inference?.p95_ms ?? '—',
    rss: r.memory?.peak_rss_mb ?? '—',
    quality: q?.composite_score,
    xling: q?.cross_lingual_pairs?.mean_cosine,
    xl_r5: q?.cross_lingual_recall_at_5 ?? q?.retrieval?.topic_cross_lang?.recall_at_5,
    r1: q?.retrieval?.recall_at_1 ?? q?.retrieval?.topic_any?.recall_at_1,
    r3: q?.retrieval?.recall_at_3 ?? q?.retrieval?.topic_any?.recall_at_3,
    r5: q?.recall_at_5 ?? q?.retrieval?.recall_at_5 ?? q?.retrieval?.topic_any?.recall_at_5,
    r10: q?.retrieval?.recall_at_10 ?? q?.retrieval?.topic_any?.recall_at_10,
    mrr10: q?.retrieval?.topic_cross_lang?.mrr_at_10 ?? q?.retrieval?.mrr_at_10,
    cohesion: q?.topic_cohesion_mean,
    separation: q?.topic_separation_mean,
    discrimination: q?.topic_discrimination,
    backend: r.backend_used ?? r.backend_requested ?? (r.status === 'ok' ? 'wasm' : '—'),
    error: r.status === 'error' ? shortError(r.error) : '—',
  };
}

async function findLatestBenchmarkJson() {
  const files = (await fs.readdir(resultsDir)).filter(
    (f) => f.startsWith('benchmark') && f.endsWith('.json'),
  );

  const ranked = [];
  for (const f of files) {
    const filePath = path.join(resultsDir, f);
    const data = await loadJson(filePath);
    const variantCount = data.results?.length ?? 0;
    if (variantCount < 15) {
      continue;
    }
    const ts = Number(f.match(/(\d+)\.json$/)?.[1] ?? 0);
    ranked.push({ filePath, ts });
  }

  ranked.sort((a, b) => b.ts - a.ts);
  return ranked[0]?.filePath ?? null;
}

async function loadJson(file) {
  return JSON.parse(await fs.readFile(file, 'utf8'));
}

function mergeResults(fullRun, gemmaRun) {
  const results = [...fullRun.results];
  const gemmaId = 'onnx-community/embeddinggemma-300m-ONNX';
  if (results.some((r) => r.model_id === gemmaId && r.status === 'ok')) {
    return results;
  }
  const withoutGemma = results.filter((r) => r.model_id !== gemmaId);
  const gemmaRows = gemmaRun?.results ?? [];
  return [...withoutGemma, ...gemmaRows];
}

function buildMarkdown(run, results) {
  const ok = results.filter((r) => r.status === 'ok');
  const peakRss = Math.max(
    run.memory_peak_rss_mb ?? 0,
    ...results.map((r) => r.memory?.peak_rss_mb ?? 0),
  );
  const ranked = [...ok].sort(
    (a, b) => (b.quality?.composite_score ?? 0) - (a.quality?.composite_score ?? 0),
  );

  const lines = [];
  lines.push('# ONNX Exploration Leaderboard');
  lines.push('');
  lines.push('Swedish/Turkish corpus (54 documents) · Transformers.js · Node.js WASM with CPU fallback');
  lines.push('');
  lines.push('## Run summary');
  lines.push('');
  lines.push(`| Field | Value |`);
  lines.push(`|-------|-------|`);
  lines.push(`| Generated | ${new Date().toISOString().slice(0, 10)} |`);
  lines.push(`| Source | \`${path.basename(run.source_file)}\`${run.gemma_source ? ` + \`${path.basename(run.gemma_source)}\`` : ''} |`);
  lines.push(`| Wall time | ${run.wall_time_human ?? formatDuration(run.wall_time_ms)} |`);
  lines.push(`| Peak RSS | ${peakRss > 0 ? peakRss.toFixed(1) : '—'} MB |`);
  lines.push(`| Documents | ${run.documents_used ?? 54} |`);
  lines.push(`| Variants tested | ${results.length} |`);
  lines.push(`| Succeeded | ${ok.length} |`);
  lines.push(`| Failed | ${results.length - ok.length} |`);
  lines.push('');
  lines.push('### Metric definitions');
  lines.push('');
  lines.push('| Metric | Description |');
  lines.push('|--------|-------------|');
  lines.push('| **Quality** | Composite: XLing (35%) + topic discrimination (25%) + XL-R@5 (25%) + R@5 (15%) |');
  lines.push('| **XLing** | Mean cosine similarity of paired SV↔TR documents (same topic) |');
  lines.push('| **XL-R@5** | Cross-lingual recall@5: query in one language, relevant doc in the *other* language in top 5 |');
  lines.push('| **R@1 / R@3 / R@5 / R@10** | Same-topic retrieval recall (any language) |');
  lines.push('| **MRR@10** | Mean reciprocal rank of first cross-lingual same-topic hit |');
  lines.push('| **Cohesion / Separation** | Mean cosine within-topic vs between-topic |');
  lines.push('| **RSS** | Peak resident set size during variant run |');
  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('## Ranked leaderboard (successful variants)');
  lines.push('');
  lines.push(
    '| Rank | Model | Quant | Backend | Quality | XLing | XL-R@5 | R@5 | R@3 | R@1 | MRR@10 | Cohesion | Sep. | ms/doc | Total | RSS MB | Dim |',
  );
  lines.push(
    '|------|-------|-------|---------|---------|-------|--------|-----|-----|-----|--------|----------|------|--------|-------|--------|-----|',
  );

  ranked.forEach((r, i) => {
    const m = rowMetrics(r);
    lines.push(
      `| ${i + 1} | ${r.model_name} | ${r.variant} | ${m.backend} | ${fmt(m.quality)} | ${fmt(m.xling)} | ${fmt(m.xl_r5)} | ${fmt(m.r5)} | ${fmt(m.r3)} | ${fmt(m.r1)} | ${fmt(m.mrr10)} | ${fmt(m.cohesion)} | ${fmt(m.separation)} | ${fmt(m.ms_doc, 1)} | ${m.total} | ${fmt(m.rss, 1)} | ${m.dim} |`,
    );
  });

  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('## Full results (all variants)');
  lines.push('');
  lines.push(
    '| Model | Quant | Status | Backend | Dim | Load | Total | ms/doc | p95 | RSS MB | Quality | XLing | XL-R@5 | R@5 | R@3 | R@1 | R@10 | Cohesion | Separation | Error |',
  );
  lines.push(
    '|-------|-------|--------|---------|-----|------|-------|--------|-----|--------|---------|-------|--------|-----|-----|-----|------|----------|------------|-------|',
  );

  const byModel = {};
  for (const r of results) {
    byModel[r.model_name] ??= [];
    byModel[r.model_name].push(r);
  }

  for (const modelName of Object.keys(byModel).sort()) {
    for (const r of byModel[modelName]) {
      const m = rowMetrics(r);
      lines.push(
        `| ${r.model_name} | ${r.variant} | ${r.status} | ${m.backend} | ${m.dim} | ${m.load_s}s | ${m.total} | ${fmt(m.ms_doc, 1)} | ${fmt(m.p95, 1)} | ${fmt(m.rss, 1)} | ${fmt(m.quality)} | ${fmt(m.xling)} | ${fmt(m.xl_r5)} | ${fmt(m.r5)} | ${fmt(m.r3)} | ${fmt(m.r1)} | ${fmt(m.r10)} | ${fmt(m.cohesion)} | ${fmt(m.separation)} | ${m.error} |`,
      );
    }
  }

  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('## Best variant per model');
  lines.push('');
  lines.push('| Model | Best quant | Quality | XLing | XL-R@5 | R@5 | R@3 | ms/doc | RSS MB | Backend |');
  lines.push('|-------|------------|---------|-------|--------|-----|-----|--------|--------|---------|');

  for (const modelName of Object.keys(byModel).sort()) {
    const modelOk = byModel[modelName].filter((r) => r.status === 'ok');
    if (modelOk.length === 0) {
      lines.push(`| ${modelName} | — | — | — | — | — | — | — | — | all failed |`);
      continue;
    }
    const best = modelOk.sort(
      (a, b) => (b.quality?.composite_score ?? 0) - (a.quality?.composite_score ?? 0),
    )[0];
    const m = rowMetrics(best);
    lines.push(
      `| ${modelName} | ${best.variant} | ${fmt(m.quality)} | ${fmt(m.xling)} | ${fmt(m.xl_r5)} | ${fmt(m.r5)} | ${fmt(m.r3)} | ${fmt(m.ms_doc, 1)} | ${fmt(m.rss, 1)} | ${m.backend} |`,
    );
  }

  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('## Failure summary');
  lines.push('');
  const failures = results.filter((r) => r.status === 'error');
  const byError = {};
  for (const r of failures) {
    const key = shortError(r.error);
    byError[key] ??= [];
    byError[key].push(`${r.model_name} / ${r.variant}`);
  }
  lines.push('| Error | Count | Variants |');
  lines.push('|-------|-------|----------|');
  for (const [err, variants] of Object.entries(byError).sort((a, b) => b[1].length - a[1].length)) {
    lines.push(`| ${err} | ${variants.length} | ${variants.join('; ')} |`);
  }

  lines.push('');
  lines.push('---');
  lines.push('');
  lines.push('*Regenerate: `npm run leaderboard`*');
  lines.push('');

  return lines.join('\n');
}

async function main() {
  const fullFile = await findLatestBenchmarkJson();

  if (!fullFile) {
    console.error('No benchmark JSON found in results/');
    process.exit(1);
  }

  const fullRun = await loadJson(fullFile);
  fullRun.source_file = fullFile;

  let gemmaRun = null;
  const hasGemma = fullRun.results?.some(
    (r) => r.model_id === 'onnx-community/embeddinggemma-300m-ONNX' && r.status === 'ok',
  );

  if (!hasGemma) {
    const allFiles = await fs.readdir(resultsDir);
    const gemmaCandidates = allFiles
      .filter((f) => f.startsWith('benchmark-') && f.endsWith('.json') && !f.includes('full'))
      .sort()
      .reverse();

    for (const f of gemmaCandidates) {
      const data = await loadJson(path.join(resultsDir, f));
      if (
        data.results?.some(
          (r) => r.model_id === 'onnx-community/embeddinggemma-300m-ONNX' && r.status === 'ok',
        )
      ) {
        gemmaRun = data;
        fullRun.gemma_source = path.join(resultsDir, f);
        break;
      }
    }
  }

  const results = filterActiveModels(mergeResults(fullRun, gemmaRun)).filter((r) => r.dtype !== 'fp16');
  const md = buildMarkdown(fullRun, results);
  await fs.writeFile(outPath, md, 'utf8');
  console.log(`Wrote ${outPath} (${results.length} variants, ${results.filter((r) => r.status === 'ok').length} ok)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

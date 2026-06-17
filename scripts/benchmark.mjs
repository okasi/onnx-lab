import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createFeatureExtractor, ModelRegistry } from '../lib/transformers-wasm.mjs';
import {
  BENCHMARK_DTYPES,
  MODELS,
  dtypeLabel,
  normalizeDtype,
} from '../config/models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    quick: false,
    models: null,
    dtypes: null,
    maxTexts: null,
    output: null,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--quick') {
      args.quick = true;
      args.maxTexts = 5;
      args.dtypes = ['q4', 'int8'];
    } else if (arg === '--max-texts') {
      args.maxTexts = Number(argv[++i]);
    } else if (arg === '--model') {
      args.models = argv[++i].split(',');
    } else if (arg === '--dtype') {
      args.dtypes = argv[++i].split(',').map(normalizeDtype);
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--help') {
      printHelp();
      process.exit(0);
    }
  }

  if (!args.dtypes) {
    args.dtypes = [...BENCHMARK_DTYPES];
  }

  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/benchmark.mjs [options]

Options:
  --quick           Small subset (2 models, 5 docs, q4+int8)
  --max-texts N     Limit corpus documents embedded per variant
  --model id[,id]   Only benchmark specific model ids
  --dtype d[,d]     Quants to test (aliases: quantized → q8)
                    Default: bnb4,fp16,int8,q4,q4f16,q8,uint8
  --output path     JSON results path (default: results/benchmark-<ts>.json)
`);
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

function mean(values) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function percentile(values, p) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((x, y) => x - y);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function round(n, digits = 2) {
  return Number(n.toFixed(digits));
}

function summarizeTimings(values) {
  if (values.length === 0) {
    return { count: 0, mean_ms: 0, p50_ms: 0, p95_ms: 0, min_ms: 0, max_ms: 0, total_ms: 0 };
  }
  const sum = values.reduce((acc, v) => acc + v, 0);
  return {
    count: values.length,
    mean_ms: round(sum / values.length),
    p50_ms: round(percentile(values, 50)),
    p95_ms: round(percentile(values, 95)),
    min_ms: round(Math.min(...values)),
    max_ms: round(Math.max(...values)),
    total_ms: round(sum),
  };
}

function snapshotMemory() {
  const m = process.memoryUsage();
  return {
    rss_mb: round(m.rss / 1024 / 1024),
    heap_used_mb: round(m.heapUsed / 1024 / 1024),
    heap_total_mb: round(m.heapTotal / 1024 / 1024),
    external_mb: round(m.external / 1024 / 1024),
    array_buffers_mb: round((m.arrayBuffers ?? 0) / 1024 / 1024),
  };
}

class MemoryMonitor {
  constructor() {
    this.peak = { rss_mb: 0, heap_used_mb: 0, external_mb: 0 };
    this.interval = null;
  }

  start() {
    this.sample();
    this.interval = setInterval(() => this.sample(), 200);
  }

  sample() {
    const s = snapshotMemory();
    for (const key of ['rss_mb', 'heap_used_mb', 'external_mb']) {
      if (s[key] > this.peak[key]) {
        this.peak[key] = s[key];
      }
    }
    return s;
  }

  stop() {
    if (this.interval) {
      clearInterval(this.interval);
      this.interval = null;
    }
    const at_end = this.sample();
    return {
      peak_rss_mb: this.peak.rss_mb,
      peak_heap_used_mb: this.peak.heap_used_mb,
      peak_external_mb: this.peak.external_mb,
      at_end,
    };
  }
}

function computeQuality(embeddings, documents, queryPairs) {
  const sameTopicSims = [];
  const diffTopicSims = [];
  const crossLangByTopic = { mortgage: [], legal: [], medical: [] };

  for (let i = 0; i < documents.length; i += 1) {
    for (let j = i + 1; j < documents.length; j += 1) {
      const a = documents[i];
      const b = documents[j];
      const va = embeddings.get(a.id);
      const vb = embeddings.get(b.id);
      if (!va || !vb) {
        continue;
      }
      const sim = cosineSimilarity(va, vb);
      if (a.topic === b.topic) {
        sameTopicSims.push(sim);
        if (a.language !== b.language) {
          crossLangByTopic[a.topic].push(sim);
        }
      } else {
        diffTopicSims.push(sim);
      }
    }
  }

  let recall1 = 0;
  let recall5 = 0;
  let recall10 = 0;
  const perTopicRetrieval = {};

  for (const query of documents) {
    const qv = embeddings.get(query.id);
    if (!qv) {
      continue;
    }

    const ranked = documents
      .filter((d) => d.id !== query.id)
      .map((d) => ({
        id: d.id,
        topic: d.topic,
        language: d.language,
        sim: cosineSimilarity(qv, embeddings.get(d.id)),
      }))
      .sort((a, b) => b.sim - a.sim);

    const sameTopicCount = ranked.filter((r) => r.topic === query.topic).length;
    if (sameTopicCount === 0) {
      continue;
    }

    const top1 = ranked[0]?.topic === query.topic;
    const top5 = ranked.slice(0, 5).some((r) => r.topic === query.topic);
    const top10 = ranked.slice(0, 10).some((r) => r.topic === query.topic);

    if (top1) {
      recall1 += 1;
    }
    if (top5) {
      recall5 += 1;
    }
    if (top10) {
      recall10 += 1;
    }

    perTopicRetrieval[query.topic] ??= { recall1: 0, recall5: 0, count: 0 };
    perTopicRetrieval[query.topic].count += 1;
    if (top1) {
      perTopicRetrieval[query.topic].recall1 += 1;
    }
    if (top5) {
      perTopicRetrieval[query.topic].recall5 += 1;
    }
  }

  const pairScores = queryPairs
    .map((pair) => {
      const sv = embeddings.get(pair.sv_doc_id);
      const tr = embeddings.get(pair.tr_doc_id);
      if (!sv || !tr) {
        return null;
      }
      return {
        pair_id: pair.id,
        topic: pair.topic,
        cosine_similarity: round(cosineSimilarity(sv, tr), 4),
      };
    })
    .filter(Boolean);

  const pairValues = pairScores.map((p) => p.cosine_similarity);
  const cohesion = mean(sameTopicSims);
  const separation = mean(diffTopicSims);

  return {
    topic_cohesion_mean: round(cohesion, 4),
    topic_separation_mean: round(separation, 4),
    topic_discrimination: round(cohesion - separation, 4),
    cross_lingual_pairs: {
      count: pairScores.length,
      mean_cosine: round(mean(pairValues), 4),
      min_cosine: pairScores.length ? round(Math.min(...pairValues), 4) : 0,
      max_cosine: pairScores.length ? round(Math.max(...pairValues), 4) : 0,
      by_topic: Object.fromEntries(
        Object.entries(crossLangByTopic).map(([topic, vals]) => [
          topic,
          { count: vals.length, mean_cosine: round(mean(vals), 4) },
        ]),
      ),
      pairs: pairScores,
    },
    retrieval: {
      recall_at_1: round(recall1 / documents.length, 4),
      recall_at_5: round(recall5 / documents.length, 4),
      recall_at_10: round(recall10 / documents.length, 4),
      by_topic: Object.fromEntries(
        Object.entries(perTopicRetrieval).map(([topic, v]) => [
          topic,
          {
            recall_at_1: round(v.recall1 / v.count, 4),
            recall_at_5: round(v.recall5 / v.count, 4),
            queries: v.count,
          },
        ]),
      ),
    },
    composite_score: round(
      mean(pairValues) * 0.4 +
        (cohesion - separation) * 0.3 +
        (recall5 / documents.length) * 0.3,
      4,
    ),
  };
}

async function loadCorpus() {
  const corpusPath = path.join(root, 'data', 'benchmark-corpus.json');
  const raw = await fs.readFile(corpusPath, 'utf8');
  return JSON.parse(raw);
}

async function listVariants(model, args) {
  const registryOptions = model.model_file_name
    ? { model_file_name: model.model_file_name }
    : {};

  let available = [];
  try {
    available = await ModelRegistry.get_available_dtypes(model.id, registryOptions);
  } catch {
    available = [];
  }

  const requested = args.dtypes.map(normalizeDtype);
  const dtypes = requested.filter((d) => available.includes(d));

  const variants = dtypes.map((dtype) => ({
    label: dtypeLabel(dtype),
    dtype,
    model_file_name: model.model_file_name ?? null,
  }));

  for (const extra of model.extra_variants ?? []) {
    if (!requested.includes(extra.dtype) && !requested.includes(extra.label)) {
      continue;
    }
    variants.push({
      label: extra.label,
      dtype: extra.dtype,
      model_file_name: extra.model_file_name,
      note: extra.note ?? null,
    });
  }

  return { variants, available_dtypes: available };
}

async function benchmarkVariant({ model, variant, documents, queryPairs }) {
  const wallStart = performance.now();
  const memoryMonitor = new MemoryMonitor();
  memoryMonitor.start();

  const result = {
    model_id: model.id,
    model_name: model.name,
    variant: variant.label,
    dtype: variant.dtype,
    model_file_name: variant.model_file_name,
    status: 'pending',
    started_at: new Date().toISOString(),
    memory_at_start: snapshotMemory(),
    load_time_ms: null,
    total_time_ms: null,
    embedding_dim: null,
    inference: null,
    quality: null,
    memory: null,
    error: null,
  };

  let extractor;
  const loadStart = performance.now();

  try {
    const options = { dtype: variant.dtype };
    if (variant.model_file_name) {
      options.model_file_name = variant.model_file_name;
    }

    extractor = await createFeatureExtractor(model.id, options);
    result.load_time_ms = round(performance.now() - loadStart);
    result.memory_after_load = snapshotMemory();

    const latencies = [];
    const embeddings = new Map();

    for (const doc of documents) {
      const t0 = performance.now();
      const tensor = await extractor(doc.text, { pooling: 'mean', normalize: true });
      const vector = tensor.tolist()[0];
      latencies.push(performance.now() - t0);
      embeddings.set(doc.id, vector);
      if (result.embedding_dim === null) {
        result.embedding_dim = vector.length;
      }
    }

    result.inference = summarizeTimings(latencies);
    result.quality = computeQuality(embeddings, documents, queryPairs);
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = error instanceof Error ? error.message : String(error);
    result.load_time_ms = round(performance.now() - loadStart);
  } finally {
    if (extractor) {
      await extractor.dispose();
    }
    if (global.gc) {
      global.gc();
    }
    result.memory = memoryMonitor.stop();
    result.total_time_ms = round(performance.now() - wallStart);
    result.finished_at = new Date().toISOString();
  }

  return result;
}

function formatDuration(ms) {
  const sec = Math.floor(ms / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  }
  if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}

function printSummaryTable(run) {
  console.log('\n' + '='.repeat(120));
  console.log('BENCHMARK SUMMARY');
  console.log('='.repeat(120));
  console.log(
    [
      'Model'.padEnd(28),
      'Quant'.padEnd(18),
      'Status'.padEnd(8),
      'Total'.padStart(8),
      'ms/doc'.padStart(8),
      'RSS MB'.padStart(8),
      'Quality'.padStart(8),
      'XLing'.padStart(8),
      'R@5'.padStart(8),
    ].join(' '),
  );
  console.log('-'.repeat(120));

  for (const r of run.results) {
    const q = r.quality;
    console.log(
      [
        r.model_name.slice(0, 27).padEnd(28),
        String(r.variant).slice(0, 17).padEnd(18),
        r.status.padEnd(8),
        (r.total_time_ms ? `${Math.round(r.total_time_ms / 1000)}s` : '-').padStart(8),
        (r.inference?.mean_ms ?? '-').toString().padStart(8),
        (r.memory?.peak_rss_mb ?? '-').toString().padStart(8),
        (q?.composite_score ?? '-').toString().padStart(8),
        (q?.cross_lingual_pairs?.mean_cosine ?? '-').toString().padStart(8),
        (q?.retrieval?.recall_at_5 ?? '-').toString().padStart(8),
      ].join(' '),
    );
  }

  console.log('-'.repeat(120));
  console.log(`Wall time: ${formatDuration(run.wall_time_ms)}`);
  console.log(`Peak RSS (run): ${run.memory_peak_rss_mb} MB`);
  console.log(`Variants: ${run.summary.ok}/${run.summary.total_variants} succeeded`);
  console.log('='.repeat(120));
}

function buildRunSummary(run) {
  const ok = run.results.filter((r) => r.status === 'ok');
  const byModel = {};

  for (const r of ok) {
    byModel[r.model_id] ??= {
      model_name: r.model_name,
      variants: [],
    };
    byModel[r.model_id].variants.push({
      variant: r.variant,
      dtype: r.dtype,
      composite_score: r.quality?.composite_score,
      cross_lingual_mean: r.quality?.cross_lingual_pairs?.mean_cosine,
      recall_at_5: r.quality?.retrieval?.recall_at_5,
      mean_ms: r.inference?.mean_ms,
      peak_rss_mb: r.memory?.peak_rss_mb,
      total_time_ms: r.total_time_ms,
    });
  }

  for (const entry of Object.values(byModel)) {
    entry.variants.sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0));
    entry.best_variant = entry.variants[0] ?? null;
  }

  return {
    total_variants: run.results.length,
    ok: ok.length,
    error: run.results.filter((r) => r.status === 'error').length,
    wall_time_ms: run.wall_time_ms,
    wall_time_human: formatDuration(run.wall_time_ms),
    memory_peak_rss_mb: run.memory_peak_rss_mb,
    dtypes_tested: [...new Set(run.results.map((r) => r.dtype))],
    best_per_model: Object.fromEntries(
      Object.entries(byModel).map(([id, v]) => [id, v.best_variant]),
    ),
    leaderboard: ok
      .map((r) => ({
        model: r.model_name,
        variant: r.variant,
        composite_score: r.quality?.composite_score,
        cross_lingual_mean: r.quality?.cross_lingual_pairs?.mean_cosine,
        recall_at_5: r.quality?.retrieval?.recall_at_5,
        mean_ms: r.inference?.mean_ms,
        peak_rss_mb: r.memory?.peak_rss_mb,
      }))
      .sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0)),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const corpus = await loadCorpus();
  const runWallStart = performance.now();
  const runMemory = new MemoryMonitor();
  runMemory.start();

  let documents = corpus.documents;
  if (args.maxTexts !== null) {
    documents = documents.slice(0, args.maxTexts);
  }

  const docIds = new Set(documents.map((d) => d.id));
  const queryPairs = corpus.query_pairs.filter(
    (p) => docIds.has(p.sv_doc_id) && docIds.has(p.tr_doc_id),
  );

  let models = MODELS;
  if (args.models) {
    models = MODELS.filter((m) => args.models.includes(m.id));
  }

  if (args.quick) {
    models = models.filter((m) =>
      ['onnx-community/bge-m3-ONNX', 'onnx-community/embeddinggemma-300m-ONNX'].includes(m.id),
    );
  }

  const run = {
    runtime: 'nodejs',
    backend: 'transformers.js-wasm',
    transformers_version: (await import('@huggingface/transformers')).env.version,
    node_version: process.version,
    args: {
      ...args,
      dtypes: args.dtypes.map((d) => dtypeLabel(d)),
    },
    dtypes_internal: args.dtypes,
    corpus_stats: corpus.stats,
    documents_used: documents.length,
    started_at: new Date().toISOString(),
    results: [],
  };

  console.log(`Benchmarking ${models.length} model(s) × [${args.dtypes.map(dtypeLabel).join(', ')}]`);
  console.log(`Documents: ${documents.length} | Backend: WASM | Node ${process.version}`);

  for (const model of models) {
    const { variants, available_dtypes } = await listVariants(model, args);
    console.log(`\n=== ${model.name} (${model.id}) ===`);
    console.log(`  Available: ${available_dtypes.map(dtypeLabel).join(', ') || 'none'}`);
    console.log(`  Testing:   ${variants.length} variant(s)`);

    for (const variant of variants) {
      process.stdout.write(`  • ${variant.label} ... `);
      const result = await benchmarkVariant({
        model,
        variant,
        documents,
        queryPairs,
      });
      run.results.push(result);

      if (result.status === 'ok') {
        console.log(
          `ok | ${formatDuration(result.total_time_ms)} | ${result.inference.mean_ms} ms/doc | ` +
            `RSS ${result.memory.peak_rss_mb} MB | quality ${result.quality.composite_score}`,
        );
      } else {
        console.log(`error: ${result.error?.slice(0, 100)}`);
      }
    }
  }

  run.finished_at = new Date().toISOString();
  run.wall_time_ms = round(performance.now() - runWallStart);
  const runMem = runMemory.stop();
  run.memory_peak_rss_mb = runMem.peak_rss_mb;
  run.memory_at_end = runMem.at_end;
  run.summary = buildRunSummary(run);

  const outDir = path.join(root, 'results');
  await fs.mkdir(outDir, { recursive: true });
  const outPath = args.output ?? path.join(outDir, `benchmark-full-${Date.now()}.json`);
  await fs.writeFile(outPath, JSON.stringify(run, null, 2), 'utf8');

  printSummaryTable(run);
  console.log(`\nWrote results to ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

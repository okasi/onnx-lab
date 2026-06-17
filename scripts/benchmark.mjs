import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createFeatureExtractor, ModelRegistry } from '../lib/transformers-wasm.mjs';
import { MODELS } from '../config/models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {
    quick: false,
    models: null,
    dtypes: null,
    maxTexts: null,
    skipFp32: true,
    output: null,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--quick') {
      args.quick = true;
      args.maxTexts = 5;
      args.skipFp32 = true;
    } else if (arg === '--include-fp32') {
      args.skipFp32 = false;
    } else if (arg === '--max-texts') {
      args.maxTexts = Number(argv[++i]);
    } else if (arg === '--model') {
      args.models = argv[++i].split(',');
    } else if (arg === '--dtype') {
      args.dtypes = argv[++i].split(',');
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--help') {
      printHelp();
      process.exit(0);
    }
  }

  return args;
}

function printHelp() {
  console.log(`Usage: node scripts/benchmark.mjs [options]

Options:
  --quick           Run a small subset (5 texts, skip fp32)
  --include-fp32    Include fp32 (downloads large .onnx_data files)
  --max-texts N     Limit corpus documents embedded per variant
  --model id[,id]   Only benchmark specific model ids
  --dtype d[,d]     Only benchmark specific dtypes
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

function percentile(values, p) {
  if (values.length === 0) {
    return 0;
  }
  const sorted = [...values].sort((x, y) => x - y);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx];
}

function summarizeTimings(values) {
  if (values.length === 0) {
    return { count: 0, mean_ms: 0, p50_ms: 0, p95_ms: 0, min_ms: 0, max_ms: 0 };
  }
  const sum = values.reduce((acc, v) => acc + v, 0);
  return {
    count: values.length,
    mean_ms: Number((sum / values.length).toFixed(2)),
    p50_ms: Number(percentile(values, 50).toFixed(2)),
    p95_ms: Number(percentile(values, 95).toFixed(2)),
    min_ms: Number(Math.min(...values).toFixed(2)),
    max_ms: Number(Math.max(...values).toFixed(2)),
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

  let dtypes = [];
  try {
    dtypes = await ModelRegistry.get_available_dtypes(model.id, registryOptions);
  } catch (error) {
    dtypes = [];
  }

  if (args.skipFp32) {
    dtypes = dtypes.filter((d) => d !== 'fp32');
  }

  if (args.dtypes) {
    dtypes = dtypes.filter((d) => args.dtypes.includes(d));
  }

  const variants = dtypes.map((dtype) => ({
    label: dtype,
    dtype,
    model_file_name: model.model_file_name ?? null,
  }));

  for (const extra of model.extra_variants ?? []) {
    if (args.dtypes && !args.dtypes.includes(extra.dtype) && !args.dtypes.includes(extra.label)) {
      continue;
    }
    if (args.skipFp32 && extra.dtype === 'fp32' && extra.label !== 'O4') {
      continue;
    }
    variants.push({
      label: extra.label,
      dtype: extra.dtype,
      model_file_name: extra.model_file_name,
      note: extra.note ?? null,
    });
  }

  return variants;
}

async function benchmarkVariant({ model, variant, documents, queryPairs }) {
  const startedAt = new Date().toISOString();
  const result = {
    model_id: model.id,
    model_name: model.name,
    variant: variant.label,
    dtype: variant.dtype,
    model_file_name: variant.model_file_name,
    status: 'pending',
    started_at: startedAt,
    load_time_ms: null,
    embedding_dim: null,
    inference: null,
    cross_lingual_pairs: null,
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
    result.load_time_ms = Number((performance.now() - loadStart).toFixed(2));

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

    const pairScores = [];
    for (const pair of queryPairs) {
      const sv = embeddings.get(pair.sv_doc_id);
      const tr = embeddings.get(pair.tr_doc_id);
      if (!sv || !tr) {
        continue;
      }
      pairScores.push({
        pair_id: pair.id,
        topic: pair.topic,
        cosine_similarity: Number(cosineSimilarity(sv, tr).toFixed(4)),
      });
    }

    if (pairScores.length > 0) {
      const scores = pairScores.map((p) => p.cosine_similarity);
      result.cross_lingual_pairs = {
        count: pairScores.length,
        mean_cosine: Number((scores.reduce((a, b) => a + b, 0) / scores.length).toFixed(4)),
        min_cosine: Number(Math.min(...scores).toFixed(4)),
        max_cosine: Number(Math.max(...scores).toFixed(4)),
        pairs: pairScores,
      };
    }

    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = error instanceof Error ? error.message : String(error);
    result.load_time_ms = Number((performance.now() - loadStart).toFixed(2));
  } finally {
    if (extractor) {
      await extractor.dispose();
    }
    if (global.gc) {
      global.gc();
    }
  }

  result.finished_at = new Date().toISOString();
  return result;
}

async function main() {
  const args = parseArgs(process.argv);
  const corpus = await loadCorpus();

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
    args.dtypes = ['q4', 'int8'];
  }

  const run = {
    runtime: 'nodejs',
    backend: 'transformers.js-wasm',
    transformers_version: (await import('@huggingface/transformers')).env.version,
    node_version: process.version,
    args,
    corpus_stats: corpus.stats,
    documents_used: documents.length,
    started_at: new Date().toISOString(),
    results: [],
  };

  console.log(`Benchmarking ${models.length} model(s) on ${documents.length} documents (WASM / Node.js)`);

  for (const model of models) {
    const variants = await listVariants(model, args);
    console.log(`\n=== ${model.name} (${model.id}) — ${variants.length} variant(s) ===`);

    for (const variant of variants) {
      process.stdout.write(`  • ${variant.label} ... `);
      const result = await benchmarkVariant({
        model,
        variant,
        documents,
        queryPairs,
      });
      run.results.push(result);
      console.log(result.status === 'ok' ? `ok (${result.inference.mean_ms} ms/doc)` : `error: ${result.error}`);
    }
  }

  run.finished_at = new Date().toISOString();
  run.summary = {
    total_variants: run.results.length,
    ok: run.results.filter((r) => r.status === 'ok').length,
    error: run.results.filter((r) => r.status === 'error').length,
  };

  const outDir = path.join(root, 'results');
  await fs.mkdir(outDir, { recursive: true });
  const outPath = args.output ?? path.join(outDir, `benchmark-${Date.now()}.json`);
  await fs.writeFile(outPath, JSON.stringify(run, null, 2), 'utf8');

  console.log(`\nWrote results to ${outPath}`);
  console.log(`Summary: ${run.summary.ok}/${run.summary.total_variants} variants succeeded`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

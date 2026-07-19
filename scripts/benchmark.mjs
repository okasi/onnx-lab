#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  BENCHMARK_DTYPES,
  MODELS,
  dtypeLabel,
  normalizeDtype,
  variantBackend,
} from '../config/models.mjs';
import {
  RESULTS_DIR,
  ROOT_DIR,
  SCRIPTS_DIR,
  formatDuration,
  loadBenchmarkCorpus,
  parseCsv,
  positiveInteger,
  round,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';
import { getModelRegistry } from '../lib/transformers-runtime.mjs';

const variantScript = path.join(SCRIPTS_DIR, 'benchmark-variant.mjs');

function parseCli() {
  const { values } = parseArgs({
    options: {
      quick: { type: 'boolean' },
      'include-fp32': { type: 'boolean' },
      'max-texts': { type: 'string' },
      model: { type: 'string' },
      dtype: { type: 'string' },
      output: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) {
    return null;
  }

  const quick = values.quick ?? false;
  const explicitDtypes = Boolean(values.dtype);
  let dtypes = parseCsv(values.dtype)?.map(normalizeDtype)
    ?? (quick ? ['q4', 'int8', 'q8'] : [...BENCHMARK_DTYPES]);
  if (values['include-fp32'] && !dtypes.includes('fp32')) {
    dtypes.push('fp32');
  }

  return {
    quick,
    models: parseCsv(values.model),
    dtypes: [...new Set(dtypes)],
    explicitDtypes: explicitDtypes || quick || Boolean(values['include-fp32']),
    maxTexts: positiveInteger(values['max-texts'] ?? (quick ? '5' : null), '--max-texts'),
    output: values.output ? path.resolve(ROOT_DIR, values.output) : null,
  };
}

function printHelp() {
  console.log(`Usage: node scripts/benchmark.mjs [options]

Options:
  --quick           EmbeddingGemma + BGE-M3, 5 docs, q4+int8+q8
  --include-fp32    Add fp32 to the requested dtypes
  --max-texts N     Limit documents per variant
  --model id[,id]   Restrict models
  --dtype d[,d]     Quants (alias: quantized -> q8)
  --output path     Results JSON path
  -h, --help        Show this help
`);
}

async function listVariants(model, args, ModelRegistry) {
  const registryOptions = model.model_file_name
    ? { model_file_name: model.model_file_name }
    : {};
  let available = [];
  let registryError = null;
  try {
    available = (await ModelRegistry.get_available_dtypes(model.id, registryOptions))
      .map(normalizeDtype);
  } catch (error) {
    registryError = error instanceof Error ? error.message : String(error);
  }

  const requested = args.dtypes.map(normalizeDtype);
  const dtypes = args.explicitDtypes
    ? requested
    : requested.filter((dtype) => available.includes(dtype));
  const variants = dtypes.map((dtype) => ({
    label: dtypeLabel(dtype),
    dtype,
    model_file_name: model.model_file_name ?? null,
    backend: variantBackend(model, { dtype }),
  }));

  for (const extra of model.extra_variants ?? []) {
    const requestedExtra = extra.when_dtype
      ? requested.includes(extra.when_dtype)
      : requested.includes(extra.dtype) || requested.includes(extra.label);
    if (!requestedExtra) {
      continue;
    }
    variants.push({
      label: extra.label,
      dtype: extra.dtype,
      model_file_name: extra.model_file_name,
      backend: variantBackend(model, extra),
    });
  }

  return { variants, available, registryError };
}

async function benchmarkVariant(model, variant, args, backend) {
  const workerArgs = [
    '--model-id', model.id,
    '--model-name', model.name,
    '--dtype', variant.dtype,
    '--variant-label', variant.label,
    '--backend', backend,
  ];
  if (variant.model_file_name) {
    workerArgs.push('--model-file-name', variant.model_file_name);
  }
  if (args.maxTexts) {
    workerArgs.push('--max-texts', String(args.maxTexts));
  }
  return (await runJsonWorker(variantScript, workerArgs, { resultFile: true })).result;
}

function printSummary(run) {
  console.log(`\n${'='.repeat(130)}`);
  console.log('BENCHMARK SUMMARY');
  console.log('='.repeat(130));
  console.log(
    [
      'Model'.padEnd(26),
      'Quant'.padEnd(16),
      'Backend'.padEnd(10),
      'Status'.padEnd(7),
      'Total'.padStart(7),
      'ms/doc'.padStart(8),
      'RSS'.padStart(7),
      'Quality'.padStart(8),
      'XLing'.padStart(7),
      'XL-R@5'.padStart(7),
      'R@5'.padStart(7),
      'R@3'.padStart(7),
      'R@1'.padStart(7),
    ].join(' '),
  );
  console.log('-'.repeat(130));

  for (const result of run.results) {
    const quality = result.quality;
    console.log(
      [
        String(result.model_name ?? '-').slice(0, 25).padEnd(26),
        String(result.variant ?? '-').slice(0, 15).padEnd(16),
        String(result.backend_used ?? result.backend_requested ?? '-').padEnd(10),
        String(result.status).padEnd(7),
        (result.total_time_ms ? `${Math.round(result.total_time_ms / 1000)}s` : '-').padStart(7),
        String(result.inference?.mean_ms ?? '-').padStart(8),
        String(result.memory?.peak_rss_mb ?? '-').padStart(7),
        String(quality?.composite_score ?? '-').padStart(8),
        String(quality?.cross_lingual_pairs?.mean_cosine ?? '-').padStart(7),
        String(quality?.cross_lingual_recall_at_5 ?? '-').padStart(7),
        String(quality?.recall_at_5 ?? '-').padStart(7),
        String(quality?.retrieval?.topic_any?.recall_at_3 ?? '-').padStart(7),
        String(quality?.retrieval?.topic_any?.recall_at_1 ?? '-').padStart(7),
      ].join(' '),
    );
  }

  console.log('-'.repeat(130));
  console.log(`Wall time: ${formatDuration(run.wall_time_ms)}`);
  console.log(`Variants: ${run.summary.ok}/${run.summary.total_variants} succeeded`);
  console.log('='.repeat(130));
}

function buildSummary(run) {
  const successful = run.results.filter((result) => result.status === 'ok');
  return {
    total_variants: run.results.length,
    ok: successful.length,
    error: run.results.length - successful.length,
    wall_time_ms: run.wall_time_ms,
    wall_time_human: formatDuration(run.wall_time_ms),
    leaderboard: successful
      .map((result) => ({
        model: result.model_name,
        variant: result.variant,
        backend: result.backend_used,
        composite_score: result.quality?.composite_score,
        cross_lingual_mean: result.quality?.cross_lingual_pairs?.mean_cosine,
        cross_lingual_r5: result.quality?.cross_lingual_recall_at_5,
        recall_at_5: result.quality?.recall_at_5,
        mean_ms: result.inference?.mean_ms,
        peak_rss_mb: result.memory?.peak_rss_mb,
      }))
      .sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0)),
  };
}

async function main() {
  const args = parseCli();
  if (!args) {
    printHelp();
    return;
  }
  const { corpus, documents } = await loadBenchmarkCorpus(args.maxTexts);
  const ModelRegistry = await getModelRegistry();
  let models = args.models
    ? MODELS.filter((model) => args.models.includes(model.id))
    : MODELS;
  if (args.models && models.length !== args.models.length) {
    const found = new Set(models.map((model) => model.id));
    throw new Error(`Unknown model(s): ${args.models.filter((id) => !found.has(id)).join(', ')}`);
  }
  if (args.quick && !args.models) {
    models = models.filter((model) =>
      ['onnx-community/bge-m3-ONNX', 'onnx-community/embeddinggemma-300m-ONNX']
        .includes(model.id));
  }

  const startedAt = performance.now();
  const transformers = await import('@huggingface/transformers');
  const run = {
    runtime: 'nodejs',
    backend: 'auto (wasm-jsep -> wasm -> cpu)',
    isolated_processes: true,
    transformers_version: transformers.env.version,
    node_version: process.version,
    args: { ...args, dtypes: args.dtypes.map(dtypeLabel) },
    corpus_stats: corpus.stats,
    documents_used: documents.length,
    started_at: new Date().toISOString(),
    results: [],
  };

  console.log(`Benchmarking ${models.length} model(s) x [${args.dtypes.map(dtypeLabel).join(', ')}]`);

  for (const model of models) {
    const { variants, available, registryError } = await listVariants(model, args, ModelRegistry);
    console.log(`\n=== ${model.name} ===`);
    console.log(`  Available: ${available.map(dtypeLabel).join(', ') || 'unknown'}`);
    if (registryError) {
      console.log(`  Registry warning: ${registryError.slice(0, 120)}`);
    }
    console.log(`  Testing:   ${variants.length} variant(s)`);

    for (const variant of variants) {
      const backends = variant.backend === 'auto'
        ? ['wasm-jsep', 'wasm', 'cpu']
        : [variant.backend];
      const failedAttempts = [];
      let result;

      for (const backend of backends) {
        process.stdout.write(`  ${variant.label} (${backend}) ... `);
        result = await benchmarkVariant(model, variant, args, backend);
        if (result.status === 'ok') {
          console.log(
            `ok | ${formatDuration(result.total_time_ms)} | quality ${result.quality.composite_score}`,
          );
          break;
        }
        failedAttempts.push({
          backend,
          error: result.error,
          error_kind: result.error_kind,
        });
        console.log(`failed: ${(result.error ?? 'unknown error').slice(0, 80)}`);
      }

      result.fallback_attempts = failedAttempts;
      result.wasm_fallback = result.status === 'ok'
        && result.backend_used === 'cpu'
        && backends[0] !== 'cpu';
      run.results.push(result);
    }
  }

  run.finished_at = new Date().toISOString();
  run.wall_time_ms = round(performance.now() - startedAt);
  run.summary = buildSummary(run);

  const outPath = args.output
    ?? path.join(
      RESULTS_DIR,
      args.quick || args.maxTexts
        ? `benchmark-${Date.now()}.json`
        : `benchmark-full-${Date.now()}.json`,
    );
  await writeJson(outPath, run);
  printSummary(run);
  console.log(`\nWrote results to ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

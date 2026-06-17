import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { getModelRegistry } from '../lib/transformers-runtime.mjs';
import {
  BENCHMARK_DTYPES,
  MODELS,
  dtypeLabel,
  normalizeDtype,
  variantBackend,
} from '../config/models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const variantScript = path.join(__dirname, 'benchmark-variant.mjs');

function parseArgs(argv) {
  const args = {
    quick: false,
    models: null,
    dtypes: null,
    maxTexts: null,
    output: null,
    isolated: true,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--quick') {
      args.quick = true;
      args.maxTexts = 5;
      args.dtypes = ['q4', 'int8', 'q8'];
    } else if (arg === '--max-texts') {
      args.maxTexts = Number(argv[++i]);
    } else if (arg === '--model') {
      args.models = argv[++i].split(',');
    } else if (arg === '--dtype') {
      args.dtypes = argv[++i].split(',').map(normalizeDtype);
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--no-isolate') {
      args.isolated = false;
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
  --quick           Small subset (includes EmbeddingGemma)
  --max-texts N     Limit documents per variant
  --model id[,id]   Restrict models
  --dtype d[,d]     Quants (alias: quantized → q8)
  --no-isolate      Run in-process (faster, OOM may crash parent)
  --output path     Results JSON path
`);
}

function round(n, digits = 2) {
  return Number(n.toFixed(digits));
}

function formatDuration(ms) {
  const sec = Math.floor(ms / 1000);
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

async function loadCorpus() {
  const raw = await fs.readFile(path.join(root, 'data', 'benchmark-corpus.json'), 'utf8');
  return JSON.parse(raw);
}

async function listVariants(model, args, ModelRegistry) {
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
    backend: variantBackend(model, { dtype }),
  }));

  for (const extra of model.extra_variants ?? []) {
    if (extra.when_dtype && !requested.includes(extra.when_dtype)) {
      continue;
    }
    if (!extra.when_dtype && !requested.includes(extra.dtype) && !requested.includes(extra.label)) {
      continue;
    }
    variants.push({
      label: extra.label,
      dtype: extra.dtype,
      model_file_name: extra.model_file_name,
      backend: variantBackend(model, extra),
      note: extra.note ?? null,
    });
  }

  return { variants, available_dtypes: available };
}

function runVariantIsolated(variantArgs) {
  return new Promise((resolve) => {
    const child = spawn(
      process.execPath,
      ['--expose-gc', variantScript, ...variantArgs],
      { stdio: ['ignore', 'pipe', 'pipe'], cwd: root },
    );

    let stderr = '';
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });

    child.on('close', async (code, signal) => {
      const resultFile = variantArgs[variantArgs.indexOf('--result-file') + 1];
      try {
        const raw = await fs.readFile(resultFile, 'utf8');
        const result = JSON.parse(raw);
        if (code !== 0 && result.status !== 'ok') {
          if (signal === 'SIGKILL' || code === 137) {
            result.status = 'error';
            result.error = result.error ?? 'Process killed (likely OOM)';
          }
        }
        resolve({ result, stderr });
      } catch (error) {
        resolve({
          result: {
            status: 'error',
            error:
              signal === 'SIGKILL' || code === 137
                ? 'Process killed (likely OOM)'
                : error instanceof Error
                  ? error.message
                  : String(error),
            stderr_tail: stderr.slice(-500),
          },
          stderr,
        });
      }
    });
  });
}

async function runVariantInProcess(variantArgs) {
  const { spawn: syncSpawn } = await import('node:child_process');
  const resultFile = variantArgs[variantArgs.indexOf('--result-file') + 1];
  await new Promise((resolve, reject) => {
    const child = syncSpawn(process.execPath, ['--expose-gc', variantScript, ...variantArgs], {
      cwd: root,
      stdio: 'inherit',
    });
    child.on('close', (code) => (code === 0 ? resolve() : reject(new Error(`exit ${code}`))));
  });
  const raw = await fs.readFile(resultFile, 'utf8');
  return { result: JSON.parse(raw), stderr: '' };
}

async function benchmarkVariant({ model, variant, args, backendOverride = null }) {
  const tmpDir = path.join(root, 'results', '.tmp');
  await fs.mkdir(tmpDir, { recursive: true });
  const resultFile = path.join(tmpDir, `variant-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);

  const variantArgs = [
    '--model-id',
    model.id,
    '--model-name',
    model.name,
    '--dtype',
    variant.dtype,
    '--variant-label',
    variant.label,
    '--backend',
    backendOverride ?? variant.backend ?? 'cpu',
    '--result-file',
    resultFile,
  ];

  if (variant.model_file_name) {
    variantArgs.push('--model-file-name', variant.model_file_name);
  }
  if (args.maxTexts !== null) {
    variantArgs.push('--max-texts', String(args.maxTexts));
  }

  const { result } = args.isolated
    ? await runVariantIsolated(variantArgs)
    : await runVariantInProcess(variantArgs);

  try {
    await fs.unlink(resultFile);
  } catch {
    // ignore
  }

  return result;
}

function printSummaryTable(run) {
  console.log('\n' + '='.repeat(130));
  console.log('BENCHMARK SUMMARY');
  console.log('='.repeat(130));
  console.log(
    [
      'Model'.padEnd(26),
      'Quant'.padEnd(16),
      'Backend'.padEnd(8),
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

  for (const r of run.results) {
    const q = r.quality;
    console.log(
      [
        r.model_name.slice(0, 25).padEnd(26),
        String(r.variant).slice(0, 15).padEnd(16),
        String(r.backend_used ?? '-').padEnd(8),
        r.status.padEnd(7),
        (r.total_time_ms ? `${Math.round(r.total_time_ms / 1000)}s` : '-').padStart(7),
        (r.inference?.mean_ms ?? '-').toString().padStart(8),
        (r.memory?.peak_rss_mb ?? '-').toString().padStart(7),
        (q?.composite_score ?? '-').toString().padStart(8),
        (q?.cross_lingual_pairs?.mean_cosine ?? '-').toString().padStart(7),
        (q?.cross_lingual_recall_at_5 ?? q?.retrieval?.topic_cross_lang?.recall_at_5 ?? '-')
          .toString()
          .padStart(7),
        (q?.recall_at_5 ?? q?.retrieval?.topic_any?.recall_at_5 ?? '-').toString().padStart(7),
        (q?.retrieval?.topic_any?.recall_at_3 ?? '-').toString().padStart(7),
        (q?.retrieval?.topic_any?.recall_at_1 ?? '-').toString().padStart(7),
      ].join(' '),
    );
  }

  console.log('-'.repeat(130));
  console.log(`Wall time: ${formatDuration(run.wall_time_ms)}`);
  console.log(`Variants: ${run.summary.ok}/${run.summary.total_variants} succeeded`);
  console.log('='.repeat(130));
}

function buildRunSummary(run) {
  const ok = run.results.filter((r) => r.status === 'ok');
  return {
    total_variants: run.results.length,
    ok: ok.length,
    error: run.results.filter((r) => r.status === 'error').length,
    wall_time_ms: run.wall_time_ms,
    wall_time_human: formatDuration(run.wall_time_ms),
    leaderboard: ok
      .map((r) => ({
        model: r.model_name,
        variant: r.variant,
        backend: r.backend_used,
        composite_score: r.quality?.composite_score,
        cross_lingual_mean: r.quality?.cross_lingual_pairs?.mean_cosine,
        cross_lingual_r5: r.quality?.cross_lingual_recall_at_5,
        recall_at_5: r.quality?.recall_at_5,
        mean_ms: r.inference?.mean_ms,
        peak_rss_mb: r.memory?.peak_rss_mb,
      }))
      .sort((a, b) => (b.composite_score ?? 0) - (a.composite_score ?? 0)),
  };
}

async function main() {
  const args = parseArgs(process.argv);
  const corpus = await loadCorpus();
  const ModelRegistry = await getModelRegistry();
  const runWallStart = performance.now();

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
    backend: 'auto (wasm → cpu fallback)',
    isolated_processes: args.isolated,
    transformers_version: (await import('@huggingface/transformers')).env.version,
    node_version: process.version,
    args: { ...args, dtypes: args.dtypes.map(dtypeLabel) },
    corpus_stats: corpus.stats,
    documents_used: args.maxTexts ?? corpus.documents.length,
    started_at: new Date().toISOString(),
    results: [],
  };

  console.log(`Benchmarking ${models.length} model(s) × [${args.dtypes.map(dtypeLabel).join(', ')}]`);
  console.log(`Isolation: ${args.isolated ? 'subprocess per variant' : 'in-process'}`);

  for (const model of models) {
    const { variants, available_dtypes } = await listVariants(model, args, ModelRegistry);
    console.log(`\n=== ${model.name} ===`);
    console.log(`  Available: ${available_dtypes.map(dtypeLabel).join(', ') || 'none'}`);
    console.log(`  Testing:   ${variants.length} variant(s)`);

    for (const variant of variants) {
      const backends =
        variant.backend === 'auto' ? ['wasm', 'cpu'] : [variant.backend ?? 'cpu'];

      let result = null;
      for (const backend of backends) {
        const suffix = backends.length > 1 ? ` (${backend})` : '';
        process.stdout.write(`  • ${variant.label}${suffix} ... `);
        result = await benchmarkVariant({ model, variant, args, backendOverride: backend });

        if (result.status === 'ok') {
          if (backend === 'cpu' && backends[0] === 'wasm') {
            result.wasm_fallback = true;
          }
          break;
        }
        if (backend === 'wasm') {
          console.log(`wasm fail → retry cpu`);
        }
      }

      run.results.push(result);

      if (result.status === 'ok') {
        console.log(
          `ok via ${result.backend_used} | ${formatDuration(result.total_time_ms)} | ` +
            `quality ${result.quality.composite_score} | XL-R@5 ${result.quality.cross_lingual_recall_at_5}`,
        );
      } else {
        console.log(`error: ${result.error?.slice(0, 90)}`);
      }
    }
  }

  run.finished_at = new Date().toISOString();
  run.wall_time_ms = round(performance.now() - runWallStart);
  run.summary = buildRunSummary(run);

  const outDir = path.join(root, 'results');
  await fs.mkdir(outDir, { recursive: true });
  const outPath = args.output ?? path.join(outDir, `benchmark-${Date.now()}.json`);
  await fs.writeFile(outPath, JSON.stringify(run, null, 2), 'utf8');

  printSummaryTable(run);
  console.log(`\nWrote results to ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

#!/usr/bin/env node
/**
 * Quality evaluation for Gemma 4 ONNX models (domain writing, JSON extract, MCQ, etc.).
 *
 * Default matrix: E2B/E4B × q4/q4f16 on CPU.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { findGemma4Model } from '../config/gemma4-models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const worker = path.join(__dirname, 'eval-gemma4-quality-worker.mjs');

const DEFAULT_VARIANTS = [
  { slug: 'E2B-it', dtype: 'q4' },
  { slug: 'E2B-it', dtype: 'q4f16' },
  { slug: 'E4B-it', dtype: 'q4' },
  { slug: 'E4B-it', dtype: 'q4f16' },
];

function parseArgs(argv) {
  const args = {
    variants: [...DEFAULT_VARIANTS],
    backend: 'cpu',
    output: null,
    category: null,
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--model') {
      const slugs = argv[++i].split(',');
      args.variants = slugs.flatMap((slug) =>
        ['q4', 'q4f16'].map((dtype) => ({ slug: slug.trim(), dtype })),
      );
    } else if (arg === '--dtype') {
      const dtypes = argv[++i].split(',');
      const slugs = args.variants.map((v) => v.slug);
      args.variants = slugs.flatMap((slug) => dtypes.map((dtype) => ({ slug, dtype: dtype.trim() })));
    } else if (arg === '--variant') {
      const [slug, dtype] = argv[++i].split(':');
      args.variants = [{ slug, dtype }];
    } else if (arg === '--backend') {
      args.backend = argv[++i];
    } else if (arg === '--category') {
      args.category = argv[++i];
    } else if (arg === '--output') {
      args.output = argv[++i];
    } else if (arg === '--help') {
      console.log(`Usage: node scripts/eval-gemma4-quality.mjs [options]

Options:
  --variant E2B-it:q4     Single variant (repeatable via multiple --variant)
  --model slug[,slug]     Filter models (default: E2B-it,E4B-it)
  --dtype d[,d]           Filter quants (default: q4,q4f16)
  --backend cpu           Backend (default cpu)
  --category NAME         Run one category only
  --output path           Results JSON path
`);
      process.exit(0);
    }
  }

  return args;
}

function runWorker(model, dtype, backend, category) {
  return new Promise((resolve) => {
    const tmpDir = path.join(root, 'results', '.tmp');
    const resultFile = path.join(tmpDir, `quality-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);
    const workerArgs = [
      '--expose-gc',
      worker,
      '--model-id', model.id,
      '--model-slug', model.slug,
      '--dtype', dtype,
      '--backend', backend,
      '--result-file', resultFile,
    ];
    if (category) workerArgs.push('--category', category);

    fs.mkdir(tmpDir, { recursive: true }).then(() => {
      const child = spawn(process.execPath, workerArgs, { cwd: root, stdio: ['ignore', 'pipe', 'pipe'] });
      let stderr = '';
      child.stderr.on('data', (d) => { stderr += d.toString(); });
      child.on('close', async () => {
        try {
          const raw = await fs.readFile(resultFile, 'utf8');
          const result = JSON.parse(raw);
          await fs.unlink(resultFile).catch(() => {});
          resolve(result);
        } catch (e) {
          resolve({
            model_slug: model.slug,
            dtype,
            status: 'error',
            error: stderr.slice(-400) || (e instanceof Error ? e.message : String(e)),
          });
        }
      });
    });
  });
}

function printSummary(run) {
  console.log('\n' + '='.repeat(100));
  console.log('GEMMA 4 QUALITY SUMMARY');
  console.log('='.repeat(100));
  console.log(
    ['Model', 'Quant', 'Overall', 'Writing', 'JSON', 'MCQ', 'Reading', 'Instruct', 'Pass%'].join('\t'),
  );
  console.log('-'.repeat(100));

  for (const r of run.results) {
    if (r.status !== 'ok') {
      console.log(`${r.model_slug}\t${r.dtype}\tFAIL\t-\t-\t-\t-\t-\t-`);
      continue;
    }
    const c = r.categories;
    console.log(
      [
        r.model_slug,
        r.dtype,
        r.overall_score,
        c.domain_writing?.summary?.mean_score ?? '-',
        c.json_extraction?.summary?.mean_score ?? '-',
        c.mcq?.summary?.mean_score ?? '-',
        c.reading_comprehension?.summary?.mean_score ?? '-',
        c.instruction_following?.summary?.mean_score ?? '-',
        r.summary?.pass_rate ?? '-',
      ].join('\t'),
    );
  }
  console.log('='.repeat(100));
}

async function writeMarkdownReport(run, outPath) {
  const lines = [
    '# Gemma 4 Quality Evaluation',
    '',
    `Tested: ${run.tested_at}`,
    `Backend: ${run.backend}`,
    `Variants: ${run.results.length}`,
    '',
    '## Overall scores',
    '',
    '| Model | Quant | Overall | Pass rate | Load (s) |',
    '|-------|-------|---------|-----------|----------|',
  ];

  for (const r of run.results) {
    if (r.status !== 'ok') {
      lines.push(`| ${r.model_slug} | ${r.dtype} | **error** | - | - |`);
      continue;
    }
    lines.push(
      `| ${r.model_slug} | ${r.dtype} | **${r.overall_score}** | ${(r.summary.pass_rate * 100).toFixed(0)}% | ${(r.load_time_ms / 1000).toFixed(1)} |`,
    );
  }

  lines.push('', '## By category', '');
  const cats = ['domain_writing', 'json_extraction', 'mcq', 'reading_comprehension', 'instruction_following'];
  const catLabels = {
    domain_writing: 'Domain writing',
    json_extraction: 'JSON extraction',
    mcq: 'MCQ (standard)',
    reading_comprehension: 'Reading comp',
    instruction_following: 'Instruction following',
  };

  for (const cat of cats) {
    lines.push(`### ${catLabels[cat]}`, '');
    lines.push('| Model | Quant | Mean score | Pass rate |');
    lines.push('|-------|-------|------------|-----------|');
    for (const r of run.results) {
      if (r.status !== 'ok') continue;
      const s = r.categories[cat]?.summary;
      if (!s) continue;
      lines.push(`| ${r.model_slug} | ${r.dtype} | ${s.mean_score} | ${(s.pass_rate * 100).toFixed(0)}% |`);
    }
    lines.push('');
  }

  lines.push('## Notes', '');
  lines.push('- Domain writing: keyword coverage, length, repetition heuristics (no LLM judge).');
  lines.push('- JSON extraction: field-level match against gold schema.');
  lines.push('- MCQ: standard single-letter multiple choice (mortgage/legal/medical).');
  lines.push('- Reading comp & instruction following: rule-based checks.');
  lines.push('');

  await fs.writeFile(outPath, lines.join('\n'));
}

async function main() {
  const args = parseArgs(process.argv);
  const variants = args.variants.map((v) => {
    const model = findGemma4Model(v.slug);
    if (!model) throw new Error(`Unknown model: ${v.slug}`);
    return { model, dtype: v.dtype };
  });

  console.log(`Gemma 4 quality eval — ${variants.length} variant(s), backend=${args.backend}\n`);

  const wallStart = performance.now();
  const results = [];

  for (const { model, dtype } of variants) {
    const label = `${model.slug} ${dtype}`;
    process.stdout.write(`→ ${label.padEnd(20)} … `);
    const result = await runWorker(model, dtype, args.backend, args.category);
    results.push(result);
    if (result.status === 'ok') {
      console.log(`overall=${result.overall_score}  pass=${(result.summary.pass_rate * 100).toFixed(0)}%`);
    } else {
      console.log(`fail: ${(result.error ?? '').slice(0, 60)}`);
    }
  }

  const run = {
    benchmark: 'gemma4-quality',
    tested_at: new Date().toISOString(),
    backend: args.backend,
    wall_time_ms: Math.round(performance.now() - wallStart),
    results,
  };

  const outPath = args.output ?? path.join(root, 'results', `eval-gemma4-quality-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(run, null, 2));

  const mdPath = path.join(root, 'GEMMA4_QUALITY.md');
  await writeMarkdownReport(run, mdPath);

  printSummary(run);
  console.log(`\nWrote ${outPath}`);
  console.log(`Wrote ${mdPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

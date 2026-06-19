#!/usr/bin/env node
/**
 * Run Gemma 4 quality suite for one model × quant (CPU).
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { round } from '../lib/gemma4-helpers.mjs';
import {
  aggregateCategory,
  overallWeightedScore,
  scoreDomainWriting,
  scoreInstructionFollowing,
  scoreJsonExtraction,
  scoreMcq,
  scoreReadingComprehension,
} from '../lib/gemma4-quality-scoring.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const TOKEN_LIMITS = {
  domain_writing: 256,
  json_extraction: 400,
  mcq: 24,
  reading_comprehension: 32,
  instruction_following: 160,
};

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--model-id') args.modelId = argv[++i];
    else if (key === '--model-slug') args.modelSlug = argv[++i];
    else if (key === '--dtype') args.dtype = argv[++i];
    else if (key === '--backend') args.backend = argv[++i];
    else if (key === '--result-file') args.resultFile = argv[++i];
    else if (key === '--category') args.category = argv[++i];
  }
  return args;
}

async function loadSuite() {
  const raw = await fs.readFile(path.join(root, 'data', 'gemma4-quality-suite.json'), 'utf8');
  return JSON.parse(raw);
}

async function generate(generator, prompt, maxNewTokens) {
  const messages = [{ role: 'user', content: prompt }];
  const out = await generator(messages, {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    temperature: 0,
    return_full_text: false,
  });
  return extractCompletion(out);
}

function extractCompletion(output) {
  const item = output?.[0];
  if (!item) return { full: '', completion: '' };

  const generated = item.generated_text;
  if (Array.isArray(generated)) {
    const assistant = generated.find((m) => m.role === 'assistant');
    const completion = (assistant?.content ?? '').trim();
    return { full: completion, completion };
  }

  if (typeof generated === 'string') {
    return { full: generated, completion: generated.trim() };
  }

  return { full: '', completion: '' };
}

async function runCategory(generator, categoryKey, categoryDef, filterCategory) {
  if (filterCategory && filterCategory !== categoryKey) {
    return { tasks: [], summary: { mean_score: 0, pass_rate: 0, count: 0 }, skipped: true };
  }

  const maxTokens = TOKEN_LIMITS[categoryKey] ?? 128;
  const taskResults = [];

  for (const task of categoryDef.tasks) {
    const t0 = performance.now();
    let generated = { full: '', completion: '' };
    let scoreResult = { score: 0, pass: false, error: null };

    try {
      generated = await generate(generator, task.prompt, maxTokens);

      switch (categoryKey) {
        case 'domain_writing':
          scoreResult = scoreDomainWriting(generated.completion || generated.full, task, categoryDef);
          break;
        case 'json_extraction':
          scoreResult = scoreJsonExtraction(generated.completion || generated.full, task);
          break;
        case 'mcq':
          scoreResult = scoreMcq(generated.completion || generated.full, task.correct);
          break;
        case 'reading_comprehension':
          scoreResult = scoreReadingComprehension(generated.completion || generated.full, task.gold_answers);
          break;
        case 'instruction_following':
          scoreResult = scoreInstructionFollowing(generated.completion || generated.full, task.rules);
          break;
        default:
          scoreResult = { score: 0, pass: false, error: `unknown category ${categoryKey}` };
      }
    } catch (e) {
      scoreResult = {
        score: 0,
        pass: false,
        error: (e instanceof Error ? e.message : String(e)).slice(0, 300),
      };
    }

    taskResults.push({
      id: task.id,
      language: task.language,
      topic: task.topic,
      score: scoreResult.score,
      pass: scoreResult.pass,
      latency_ms: round(performance.now() - t0),
      completion: (generated.completion || generated.full).slice(0, 1200),
      metrics: scoreResult,
    });
  }

  return {
    tasks: taskResults,
    summary: aggregateCategory(taskResults),
    weight: categoryDef.weight,
  };
}

async function main() {
  const args = parseArgs(process.argv);
  if (!args.modelId || !args.dtype || !args.resultFile) {
    console.error('usage: --model-id ID --model-slug SLUG --dtype q4 --result-file path [--backend cpu]');
    process.exit(1);
  }

  const suite = await loadSuite();
  const backend = args.backend ?? 'cpu';
  const result = {
    model_id: args.modelId,
    model_slug: args.modelSlug,
    dtype: args.dtype,
    backend,
    status: 'pending',
    load_time_ms: null,
    categories: {},
    overall_score: 0,
    tested_at: new Date().toISOString(),
  };

  let generator;
  try {
    const loadStart = performance.now();
    generator = await createTextGenerator(
      args.modelId,
      {
        dtype: args.dtype,
        session_options: {
          enableCpuMemArena: false,
          enableMemPattern: false,
        },
      },
      backend,
    );
    result.load_time_ms = round(performance.now() - loadStart);
    result.status = 'ok';

    const weights = {};
    for (const [key, def] of Object.entries(suite.categories)) {
      weights[key] = def.weight;
      result.categories[key] = await runCategory(generator, key, def, args.category);
    }

    const summaries = Object.fromEntries(
      Object.entries(result.categories).map(([k, v]) => [k, v.summary]),
    );
    result.overall_score = overallWeightedScore(summaries, weights);
    result.summary = {
      pass_rate: round(
        Object.values(result.categories).flatMap((c) => c.tasks).filter((t) => t.pass).length
        / Math.max(1, Object.values(result.categories).flatMap((c) => c.tasks).length),
      ),
      task_count: Object.values(result.categories).reduce((n, c) => n + c.tasks.length, 0),
    };
  } catch (e) {
    result.status = 'error';
    result.error = (e instanceof Error ? e.message : String(e)).slice(0, 500);
  } finally {
    if (generator) {
      try {
        await generator.dispose();
      } catch {
        // ignore
      }
    }
  }

  await fs.writeFile(args.resultFile, JSON.stringify(result, null, 2));
  console.log(JSON.stringify({
    model_slug: result.model_slug,
    dtype: result.dtype,
    status: result.status,
    overall_score: result.overall_score,
    load_time_ms: result.load_time_ms,
  }));
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

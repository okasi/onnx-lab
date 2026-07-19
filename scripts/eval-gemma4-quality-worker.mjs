#!/usr/bin/env node
import { parseArgs } from 'node:util';
import {
  dispose,
  positiveInteger,
  projectPath,
  readJson,
  round,
  writeJson,
} from '../lib/benchmark-support.mjs';
import {
  aggregateCategory,
  overallWeightedScore,
  scoreClassification,
  scoreDomainWriting,
  scoreInstructionFollowing,
  scoreJsonExtraction,
  scoreMcq,
  scoreReadingComprehension,
  scoreSummarization,
} from '../lib/gemma4-quality-scoring.mjs';
import { createTextGenerator } from '../lib/transformers-runtime.mjs';

const TOKEN_LIMITS = {
  domain_writing: 300,
  json_extraction: 512,
  mcq: 32,
  reading_comprehension: 48,
  instruction_following: 200,
  summarization: 180,
  classification: 16,
};

const SCORERS = {
  domain_writing: (text, task, config) => scoreDomainWriting(text, task, config),
  json_extraction: (text, task) => scoreJsonExtraction(text, task),
  mcq: (text, task) => scoreMcq(text, task.correct),
  reading_comprehension: (text, task) =>
    scoreReadingComprehension(text, task.gold_answers),
  instruction_following: (text, task) =>
    scoreInstructionFollowing(text, task.rules),
  summarization: (text, task) => scoreSummarization(text, task),
  classification: (text, task) => scoreClassification(text, task.expected),
};

function parseCli() {
  const { values } = parseArgs({
    options: {
      'model-id': { type: 'string' },
      'model-slug': { type: 'string' },
      dtype: { type: 'string' },
      backend: { type: 'string' },
      category: { type: 'string' },
      'max-tasks': { type: 'string' },
      'result-file': { type: 'string' },
    },
    strict: true,
  });
  for (const required of ['model-id', 'model-slug', 'dtype', 'result-file']) {
    if (!values[required]) throw new Error(`Missing --${required}`);
  }
  return {
    modelId: values['model-id'],
    modelSlug: values['model-slug'],
    dtype: values.dtype,
    backend: values.backend ?? 'cpu',
    category: values.category ?? null,
    maxTasks: positiveInteger(values['max-tasks'], '--max-tasks'),
    resultFile: values['result-file'],
  };
}

async function generate(generator, prompt, maxNewTokens) {
  const output = await generator([{ role: 'user', content: prompt }], {
    max_new_tokens: maxNewTokens,
    do_sample: false,
    return_full_text: false,
  });
  const generated = output?.[0]?.generated_text;
  if (Array.isArray(generated)) {
    const assistant = generated.filter((message) => message.role === 'assistant').at(-1);
    return String(assistant?.content ?? '').trim();
  }
  return typeof generated === 'string' ? generated.trim() : '';
}

async function runCategory(generator, categoryKey, category, maxTasks) {
  const scorer = SCORERS[categoryKey];
  if (!scorer) {
    throw new Error(`Unknown category: ${categoryKey}`);
  }
  const taskResults = [];
  for (const task of category.tasks.slice(0, maxTasks ?? category.tasks.length)) {
    const startedAt = performance.now();
    let completion = '';
    let metrics;
    try {
      completion = await generate(
        generator,
        task.prompt,
        TOKEN_LIMITS[categoryKey] ?? 128,
      );
      metrics = scorer(completion, task, category);
    } catch (error) {
      metrics = {
        score: 0,
        pass: false,
        error: (error instanceof Error ? error.message : String(error)).slice(0, 300),
      };
    }
    taskResults.push({
      id: task.id,
      language: task.language,
      topic: task.topic,
      score: metrics.score,
      pass: metrics.pass,
      latency_ms: round(performance.now() - startedAt),
      completion: completion.slice(0, 1200),
      metrics,
    });
  }
  return {
    tasks: taskResults,
    summary: aggregateCategory(taskResults),
    weight: category.weight,
  };
}

async function main() {
  const args = parseCli();
  const suite = await readJson(projectPath('data', 'gemma4-quality-suite.json'));
  if (args.category && !suite.categories[args.category]) {
    throw new Error(`Unknown category: ${args.category}`);
  }
  const selectedCategories = Object.entries(suite.categories)
    .filter(([key]) => !args.category || key === args.category);
  const result = {
    model_id: args.modelId,
    model_slug: args.modelSlug,
    dtype: args.dtype,
    backend: args.backend,
    status: 'pending',
    load_time_ms: null,
    categories: {},
    overall_score: 0,
    tested_at: new Date().toISOString(),
  };

  let generator;
  try {
    const loadStartedAt = performance.now();
    generator = await createTextGenerator(
      args.modelId,
      {
        dtype: args.dtype,
        session_options: {
          enableCpuMemArena: false,
          enableMemPattern: false,
        },
      },
      args.backend,
    );
    result.load_time_ms = round(performance.now() - loadStartedAt);

    const weights = {};
    for (const [key, category] of selectedCategories) {
      weights[key] = category.weight;
      result.categories[key] = await runCategory(
        generator,
        key,
        category,
        args.maxTasks,
      );
    }
    const summaries = Object.fromEntries(
      Object.entries(result.categories).map(([key, category]) => [key, category.summary]),
    );
    const tasks = Object.values(result.categories).flatMap((category) => category.tasks);
    result.overall_score = overallWeightedScore(summaries, weights);
    result.summary = {
      pass_rate: round(tasks.filter((task) => task.pass).length / Math.max(1, tasks.length)),
      task_count: tasks.length,
    };
    result.status = 'ok';
  } catch (error) {
    result.status = 'error';
    result.error = (error instanceof Error ? error.message : String(error)).slice(0, 500);
  } finally {
    await dispose(generator);
    await writeJson(args.resultFile, result);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

#!/usr/bin/env node
import { parseArgs } from 'node:util';
import { findGemma4Model } from '../config/gemma4-models.mjs';
import {
  MemoryMonitor,
  classifyRuntimeError,
  dispose,
  positiveInteger,
  projectPath,
  readJson,
  round,
  writeJson,
} from '../lib/benchmark-support.mjs';
import {
  loadGemma4Multimodal,
  runGemma4MultimodalTask,
} from '../lib/gemma4-multimodal-runtime.mjs';
import { scoreMultimodalOutput } from '../lib/gemma4-multimodal-scoring.mjs';
import { loadMultimodalTasks } from '../lib/gemma4-multimodal-suite.mjs';

function parseCli() {
  const { values } = parseArgs({
    options: {
      'model-slug': { type: 'string' },
      dtype: { type: 'string' },
      modality: { type: 'string' },
      'max-tasks': { type: 'string' },
      'result-file': { type: 'string' },
    },
    strict: true,
  });
  for (const required of ['model-slug', 'dtype', 'result-file']) {
    if (!values[required]) throw new Error(`Missing --${required}`);
  }
  if (values.modality && !['image', 'audio'].includes(values.modality)) {
    throw new Error('--modality must be image or audio');
  }
  return {
    modelSlug: values['model-slug'],
    dtype: values.dtype,
    modality: values.modality ?? null,
    maxTasks: positiveInteger(values['max-tasks'], '--max-tasks'),
    resultFile: values['result-file'],
  };
}

async function main() {
  const args = parseCli();
  const modelSpec = findGemma4Model(args.modelSlug);
  const result = {
    model_slug: modelSpec?.slug ?? args.modelSlug,
    model_id: modelSpec?.id ?? null,
    dtype: args.dtype,
    modality: args.modality ?? 'all',
    backend: 'cpu',
    status: 'pending',
    load_time_ms: null,
    tasks: [],
    summary: { total: 0, ok: 0, pass: 0 },
    memory: null,
    error: null,
    error_kind: null,
    started_at: new Date().toISOString(),
  };
  if (!modelSpec) {
    result.status = 'error';
    result.error = `Unknown model slug: ${args.modelSlug}`;
    await writeJson(args.resultFile, result);
    return;
  }

  const suite = await readJson(projectPath('data', 'gemma4-multimodal-suite.json'));
  const tasks = loadMultimodalTasks(suite, args.modality)
    .slice(0, args.maxTasks ?? Number.POSITIVE_INFINITY);
  if (!tasks.length) {
    throw new Error('No multimodal tasks selected');
  }

  const monitor = new MemoryMonitor();
  monitor.start();
  let generationModel;
  try {
    const loadStartedAt = performance.now();
    const loaded = await loadGemma4Multimodal(modelSpec.id, args.dtype);
    const { processor } = loaded;
    generationModel = loaded.model;
    result.load_time_ms = round(performance.now() - loadStartedAt);
    global.gc?.();

    for (const task of tasks) {
      const taskResult = {
        id: task.id,
        modality: task.modality,
        status: 'pending',
        infer_ms: null,
        generated_text: null,
        score: null,
        pass: false,
        error: null,
      };
      try {
        const startedAt = performance.now();
        const output = await runGemma4MultimodalTask({
          processor,
          model: generationModel,
          modality: task.modality,
          promptText: task.prompt,
          mediaUrl: task.media_url,
          maxNewTokens: task.max_new_tokens,
        });
        taskResult.infer_ms = round(performance.now() - startedAt);
        taskResult.generated_text = output.generated_text;
        const scored = scoreMultimodalOutput(
          output.generated_text,
          task.expect_substrings ?? [],
        );
        taskResult.score = round(scored.score, 3);
        taskResult.pass = scored.pass;
        taskResult.matched = scored.matched;
        taskResult.missing = scored.missing;
        taskResult.status = 'ok';
      } catch (error) {
        taskResult.status = 'error';
        taskResult.error = (error instanceof Error ? error.message : String(error))
          .slice(0, 500);
      }
      result.tasks.push(taskResult);
    }

    result.summary = {
      total: result.tasks.length,
      ok: result.tasks.filter((task) => task.status === 'ok').length,
      pass: result.tasks.filter((task) => task.pass).length,
    };
    result.status = result.summary.ok === result.summary.total ? 'ok' : 'infer_error';
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    result.error = message.slice(0, 600);
    result.error_kind = classifyRuntimeError(message);
    result.status = result.load_time_ms == null ? 'error' : 'infer_error';
  } finally {
    await dispose(generationModel);
    result.memory = monitor.stop();
    result.finished_at = new Date().toISOString();
    await writeJson(args.resultFile, result);
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

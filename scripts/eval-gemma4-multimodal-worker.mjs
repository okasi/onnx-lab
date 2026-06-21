#!/usr/bin/env node
/**
 * Run Gemma 4 multimodal eval for one model × quant (isolated process).
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { findGemma4Model } from '../config/gemma4-models.mjs';
import { MemoryMonitor, classifyGemma4Error, round } from '../lib/gemma4-helpers.mjs';
import { loadGemma4Multimodal, runGemma4MultimodalTask } from '../lib/gemma4-multimodal-runtime.mjs';
import { scoreMultimodalOutput } from '../lib/gemma4-multimodal-scoring.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function parseArgs(argv) {
  const args = {};
  for (let i = 2; i < argv.length; i += 1) {
    const key = argv[i];
    if (key === '--model-slug') args.modelSlug = argv[++i];
    else if (key === '--dtype') args.dtype = argv[++i];
    else if (key === '--result-file') args.resultFile = argv[++i];
  }
  return args;
}

async function main() {
  const args = parseArgs(process.argv);
  const model = findGemma4Model(args.modelSlug);
  const suite = JSON.parse(
    await fs.readFile(path.join(root, 'data', 'gemma4-multimodal-suite.json'), 'utf8'),
  );

  const result = {
    model_slug: model?.slug ?? args.modelSlug,
    model_id: model?.id ?? null,
    dtype: args.dtype,
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

  if (!model) {
    result.status = 'error';
    result.error = `Unknown model slug: ${args.modelSlug}`;
    await fs.writeFile(args.resultFile, JSON.stringify(result, null, 2));
    return;
  }

  const monitor = new MemoryMonitor();
  monitor.start();

  try {
    const loadStart = performance.now();
    const { processor, model: genModel } = await loadGemma4Multimodal(model.id, args.dtype);
    result.load_time_ms = round(performance.now() - loadStart);

    if (global.gc) global.gc();

    for (const task of suite.tasks) {
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
        const t0 = performance.now();
        const out = await runGemma4MultimodalTask({
          processor,
          model: genModel,
          modality: task.modality,
          promptText: task.prompt,
          mediaUrl: task.media_url,
          maxNewTokens: task.max_new_tokens,
        });
        taskResult.infer_ms = round(performance.now() - t0);
        taskResult.generated_text = out.generated_text;
        const scored = scoreMultimodalOutput(out.generated_text, task.expect_substrings ?? []);
        taskResult.score = round(scored.score, 3);
        taskResult.pass = scored.pass;
        taskResult.matched = scored.matched;
        taskResult.missing = scored.missing;
        taskResult.status = 'ok';
      } catch (error) {
        const msg = error instanceof Error ? error.message : String(error);
        taskResult.status = 'error';
        taskResult.error = msg.slice(0, 500);
      }

      result.tasks.push(taskResult);
    }

    result.summary.total = result.tasks.length;
    result.summary.ok = result.tasks.filter((t) => t.status === 'ok').length;
    result.summary.pass = result.tasks.filter((t) => t.pass).length;
    result.status = result.summary.ok === result.summary.total ? 'ok' : 'infer_error';
  } catch (error) {
    const msg = error instanceof Error ? error.message : String(error);
    result.error = msg.slice(0, 600);
    result.error_kind = classifyGemma4Error(msg);
    result.status = result.load_time_ms != null ? 'infer_error' : 'error';
  } finally {
    result.memory = monitor.stop();
    result.finished_at = new Date().toISOString();
    await fs.writeFile(args.resultFile, JSON.stringify(result, null, 2));
  }
}

main().catch(async (error) => {
  console.error(error);
  process.exit(1);
});

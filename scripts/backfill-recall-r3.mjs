#!/usr/bin/env node
/**
 * Re-runs successful benchmark variants to add recall_at_3 to stored JSON results.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import { MODELS } from '../config/models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const resultsDir = path.join(root, 'results');
const variantScript = path.join(__dirname, 'benchmark-variant.mjs');

function needsRecallAt3(result) {
  if (result.status !== 'ok' || !result.quality?.retrieval) {
    return false;
  }
  const r = result.quality.retrieval;
  return r.recall_at_3 === undefined && r.topic_any?.recall_at_3 === undefined;
}

function patchRecallAt3(target, source) {
  const sourceRetrieval = source?.quality?.retrieval;
  const targetRetrieval = target?.quality?.retrieval;
  if (!sourceRetrieval || !targetRetrieval) {
    return false;
  }

  const sourceTasks = sourceRetrieval.topic_any
    ? sourceRetrieval
    : { topic_any: sourceRetrieval };

  let patched = false;

  for (const [task, metrics] of Object.entries(sourceTasks)) {
    if (metrics.recall_at_3 === undefined) {
      continue;
    }

    if (targetRetrieval[task]) {
      targetRetrieval[task].recall_at_3 = metrics.recall_at_3;
      for (const [topic, byTopic] of Object.entries(metrics.by_topic ?? {})) {
        if (byTopic.recall_at_3 === undefined) {
          continue;
        }
        targetRetrieval[task].by_topic ??= {};
        targetRetrieval[task].by_topic[topic] ??= {};
        targetRetrieval[task].by_topic[topic].recall_at_3 = byTopic.recall_at_3;
      }
      patched = true;
      continue;
    }

    if (task === 'topic_any' && targetRetrieval.recall_at_5 !== undefined) {
      targetRetrieval.recall_at_3 = metrics.recall_at_3;
      for (const [topic, byTopic] of Object.entries(metrics.by_topic ?? {})) {
        if (byTopic.recall_at_3 === undefined) {
          continue;
        }
        targetRetrieval.by_topic ??= {};
        targetRetrieval.by_topic[topic] ??= {};
        targetRetrieval.by_topic[topic].recall_at_3 = byTopic.recall_at_3;
      }
      patched = true;
    }
  }

  return patched;
}

function backendsFor(result, model) {
  if (result.backend_used === 'wasm' || result.backend_used === 'cpu') {
    return [result.backend_used];
  }
  const requested = model.backend ?? 'auto';
  return requested === 'auto' ? ['wasm', 'cpu'] : [requested];
}

async function runVariant(result) {
  const model = MODELS.find((m) => m.id === result.model_id);
  if (!model) {
    throw new Error(`Unknown model_id: ${result.model_id}`);
  }

  const tmpDir = path.join(resultsDir, '.tmp');
  const resultFile = path.join(tmpDir, `backfill-${Date.now()}-${Math.random().toString(36).slice(2)}.json`);

  const baseArgs = [
    '--model-id',
    result.model_id,
    '--model-name',
    result.model_name,
    '--dtype',
    result.dtype,
    '--variant-label',
    result.variant,
    '--result-file',
    resultFile,
  ];

  if (result.model_file_name) {
    baseArgs.push('--model-file-name', result.model_file_name);
  }

  let lastError = null;
  for (const backend of backendsFor(result, model)) {
    const variantArgs = [...baseArgs, '--backend', backend];
    try {
      const fresh = await runVariantOnce(variantArgs, resultFile);
      if (fresh.status === 'ok') {
        return fresh;
      }
      lastError = new Error(fresh.error ?? `failed on ${backend}`);
    } catch (error) {
      lastError = error;
      if (backend === 'wasm') {
        continue;
      }
    }
  }

  throw lastError ?? new Error('backfill failed');
}

function runVariantOnce(variantArgs, resultFile) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, ['--expose-gc', variantScript, ...variantArgs], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    child.stderr.on('data', () => {});

    child.on('close', async (code) => {
      try {
        const fresh = JSON.parse(await fs.readFile(resultFile, 'utf8'));
        await fs.unlink(resultFile).catch(() => {});
        if (fresh.status !== 'ok') {
          reject(new Error(fresh.error ?? `exit ${code}`));
          return;
        }
        resolve(fresh);
      } catch (error) {
        reject(error);
      }
    });
  });
}

async function backfillFile(filePath) {
  const data = JSON.parse(await fs.readFile(filePath, 'utf8'));
  const pending = data.results.filter(needsRecallAt3);
  if (pending.length === 0) {
    console.log(`${path.basename(filePath)}: already has R@3`);
    return 0;
  }

  console.log(`${path.basename(filePath)}: backfilling ${pending.length} variants...`);
  let updated = 0;

  for (const result of pending) {
    process.stdout.write(`  ${result.model_name} / ${result.variant} ... `);
    try {
      const fresh = await runVariant(result);
      if (patchRecallAt3(result, fresh)) {
        updated += 1;
        const r3 =
          fresh.quality.retrieval.topic_any?.recall_at_3 ?? fresh.quality.retrieval.recall_at_3;
        console.log(`ok (R@3=${r3})`);
      } else {
        console.log('skip (no retrieval data)');
      }
    } catch (error) {
      console.log(`error: ${error instanceof Error ? error.message : error}`);
    }
  }

  if (updated > 0) {
    await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    console.log(`  saved ${updated} updates to ${path.basename(filePath)}`);
  }

  return updated;
}

async function main() {
  await fs.mkdir(path.join(resultsDir, '.tmp'), { recursive: true });
  const candidates = (await fs.readdir(resultsDir))
    .filter((f) => f.startsWith('benchmark') && f.endsWith('.json'))
    .sort();

  const files = candidates.filter(
    (f) => f.startsWith('benchmark-full-') || f === 'benchmark-1781707883726.json',
  );

  let total = 0;
  for (const file of files) {
    total += await backfillFile(path.join(resultsDir, file));
  }

  console.log(`Done. Patched ${total} variant(s). Run: npm run leaderboard`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

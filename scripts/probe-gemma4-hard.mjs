#!/usr/bin/env node
import path from 'node:path';
import { parseArgs } from 'node:util';
import {
  RESULTS_DIR,
  SCRIPTS_DIR,
  parseCsv,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';
import { GEMMA4_WEBGPU_STRATEGIES } from '../lib/gemma4-webgpu.mjs';

const strategyScript = path.join(SCRIPTS_DIR, 'probe-gemma4-webgpu-strategy.mjs');
const DEFAULT_MODEL = 'onnx-community/gemma-4-E2B-it-ONNX';
const DEFAULT_STRATEGIES = Object.keys(GEMMA4_WEBGPU_STRATEGIES);

function parseCli() {
  const { values } = parseArgs({
    options: {
      model: { type: 'string' },
      strategies: { type: 'string' },
      help: { type: 'boolean', short: 'h' },
    },
    strict: true,
  });
  if (values.help) return null;
  const strategies = parseCsv(values.strategies) ?? DEFAULT_STRATEGIES;
  const unknown = strategies.filter((name) => !GEMMA4_WEBGPU_STRATEGIES[name]);
  if (unknown.length) throw new Error(`Unknown strategies: ${unknown.join(', ')}`);
  return { modelId: values.model ?? DEFAULT_MODEL, strategies };
}

async function main() {
  const args = parseCli();
  if (!args) {
    console.log('Usage: node scripts/probe-gemma4-hard.mjs [--model ID] [--strategies a,b]');
    return;
  }
  console.log(`Gemma 4 WebGPU hard probe - ${args.modelId}\n`);
  const results = [];
  for (const strategy of args.strategies) {
    process.stdout.write(`${strategy} ... `);
    const result = (await runJsonWorker(
      strategyScript,
      [args.modelId, strategy],
      { exposeGc: false, resultFile: false },
    )).result;
    results.push(result);
    console.log(
      result.status === 'ok'
        ? `ok load=${result.load_ms}ms infer=${result.infer_ms}ms f16=${result.shader_f16}`
        : `failed: ${(result.error ?? '').slice(0, 80)}`,
    );
  }

  const successful = results.filter((result) => result.status === 'ok');
  const report = {
    tested_at: new Date().toISOString(),
    model_id: args.modelId,
    results,
    summary: {
      succeeded: successful.length,
      total: results.length,
      best_infer_ms: successful.length
        ? Math.min(...successful.map((result) => result.infer_ms))
        : null,
      shader_f16_seen: results.some((result) => result.shader_f16 === true),
    },
  };
  const outPath = path.join(RESULTS_DIR, `probe-gemma4-webgpu-hard-${Date.now()}.json`);
  await writeJson(outPath, report);
  console.log(`\n${successful.length}/${results.length} strategies succeeded`);
  console.log(`Wrote ${outPath}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

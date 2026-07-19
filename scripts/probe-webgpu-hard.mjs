#!/usr/bin/env node
import path from 'node:path';
import {
  RESULTS_DIR,
  SCRIPTS_DIR,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';

const strategyScript = path.join(SCRIPTS_DIR, 'probe-webgpu-browser-strategy.mjs');
const STRATEGIES = [
  'browser:q4-control',
  'browser:default',
  'browser:force-cpu-gather',
  'browser:webgpu-wasm-dual',
  'browser:angle-gl',
  'browser:vulkan-angle',
  'browser:lavapipe-vk',
  'browser:chrome-blog-flags',
];

function summarize(results) {
  const successful = results.filter((result) => result.status === 'ok');
  const q4 = successful.find((result) => result.strategy === 'browser:q4-control');
  const q4f16 = results.filter((result) => result.strategy !== 'browser:q4-control');
  return {
    webgpu_q4_works: Boolean(q4),
    webgpu_q4_infer_ms: q4?.infer_ms ?? null,
    q4f16_shader_f16_available: q4f16.some((result) => result.shader_f16 === true),
    q4f16_any_ok: q4f16.some((result) => result.status === 'ok'),
    succeeded: successful.length,
    total: results.length,
  };
}

function printSummary(summary) {
  console.log('\n--- Summary ---');
  console.log(`WebGPU + q4: ${summary.webgpu_q4_works ? 'works' : 'failed'}`);
  console.log(`WebGPU + q4f16: ${summary.q4f16_any_ok ? 'works' : 'blocked'}`);
  console.log(`shader-f16 seen: ${summary.q4f16_shader_f16_available}`);
  console.log(`${summary.succeeded}/${summary.total} strategies succeeded`);
}

async function main() {
  console.log('EmbeddingGemma WebGPU hard probe\n');
  const results = [];
  for (const strategy of STRATEGIES) {
    process.stdout.write(`${strategy} ... `);
    const result = (await runJsonWorker(
      strategyScript,
      [strategy],
      { exposeGc: false, resultFile: false },
    )).result;
    results.push(result);
    console.log(
      result.status === 'ok'
        ? `ok infer=${result.infer_ms}ms f16=${result.shader_f16}`
        : `failed: ${(result.error ?? '').slice(0, 90)}`,
    );
  }
  const summary = summarize(results);
  const outPath = path.join(RESULTS_DIR, `probe-webgpu-hard-${Date.now()}.json`);
  await writeJson(outPath, { results, summary });
  console.log(`\nWrote ${outPath}`);
  printSummary(summary);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

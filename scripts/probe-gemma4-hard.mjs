#!/usr/bin/env node
/**
 * Hard WebGPU probe for Gemma 4 — tries multiple Chrome strategies in isolated subprocesses.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const strategyScript = path.join(__dirname, 'probe-gemma4-webgpu-strategy.mjs');

const DEFAULT_MODEL = 'onnx-community/gemma-4-E2B-it-ONNX';
const STRATEGIES = [
  'browser:q4-control',
  'browser:q4-force-gather',
  'browser:q4-dual-ep',
  'browser:q4f16',
  'browser:q8',
  'browser:angle-gl',
  'browser:vulkan-angle',
];

function argValue(flag) {
  const i = process.argv.indexOf(flag);
  return i === -1 ? null : process.argv[i + 1];
}

async function runStrategy(modelId, strategy) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [strategyScript, modelId, strategy], {
      cwd: root,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d; });
    child.stderr.on('data', (d) => { stderr += d; });
    child.on('close', () => {
      try {
        resolve(JSON.parse(stdout.trim()));
      } catch {
        resolve({
          strategy,
          model_id: modelId,
          status: 'error',
          error: stderr.trim() || stdout.trim() || 'parse error',
        });
      }
    });
  });
}

async function main() {
  const modelId = argValue('--model') ?? DEFAULT_MODEL;
  const strategies = argValue('--strategies')?.split(',') ?? STRATEGIES;
  console.log(`Gemma 4 WebGPU hard probe — ${modelId}\n`);

  const results = [];
  for (const strategy of strategies) {
    process.stdout.write(`→ ${strategy} … `);
    const r = await runStrategy(modelId, strategy);
    results.push(r);
    if (r.status === 'ok') {
      console.log(`ok load=${r.load_ms}ms infer=${r.infer_ms}ms f16=${r.shader_f16}`);
    } else {
      console.log(`fail: ${(r.error ?? '').slice(0, 80)}`);
    }
  }

  const ok = results.filter((r) => r.status === 'ok');
  const report = {
    tested_at: new Date().toISOString(),
    model_id: modelId,
    results,
    summary: {
      succeeded: ok.length,
      total: results.length,
      best_infer_ms: ok.length ? Math.min(...ok.map((r) => r.infer_ms)) : null,
      shader_f16_seen: results.some((r) => r.shader_f16 === true),
    },
  };

  const outPath = path.join(root, 'results', `probe-gemma4-webgpu-hard-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(report, null, 2));
  console.log(`\n${ok.length}/${results.length} strategies succeeded`);
  console.log(`Wrote ${outPath}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

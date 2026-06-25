#!/usr/bin/env node
/**
 * Comprehensive WebGPU probe for EmbeddingGemma 300M.
 * Runs browser strategies in isolated subprocesses; writes JSON report.
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const strategyScript = path.join(__dirname, 'probe-webgpu-browser-strategy.mjs');

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

async function runStrategy(name) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, [strategyScript, name], {
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
        resolve({ strategy: name, status: 'error', error: stderr.trim() || stdout.trim() || 'parse error' });
      }
    });
  });
}

async function main() {
  console.log('EmbeddingGemma WebGPU hard probe\n');
  const results = [];

  for (const name of STRATEGIES) {
    process.stdout.write(`→ ${name} … `);
    const r = await runStrategy(name);
    results.push(r);
    if (r.status === 'ok') {
      console.log(`✓ dtype implied infer=${r.infer_ms}ms f16=${r.shader_f16}`);
    } else {
      console.log(`✗ ${(r.error ?? '').slice(0, 90)}`);
    }
  }

  const outPath = path.join(root, 'results', `probe-webgpu-hard-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify({ results, summary: summarize(results) }, null, 2));
  console.log(`\nWrote ${outPath}`);
  printSummary(summarize(results));
}

function summarize(results) {
  const ok = results.filter((r) => r.status === 'ok');
  const q4 = ok.find((r) => r.strategy === 'browser:q4-control');
  const q4f16Attempts = results.filter((r) => r.strategy !== 'browser:q4-control');
  return {
    webgpu_q4_works: Boolean(q4),
    webgpu_q4_infer_ms: q4?.infer_ms ?? null,
    q4f16_shader_f16_available: q4f16Attempts.some((r) => r.shader_f16 === true),
    q4f16_any_ok: q4f16Attempts.some((r) => r.status === 'ok'),
    succeeded: ok.length,
    total: results.length,
  };
}

function printSummary(s) {
  console.log('\n--- Summary ---');
  console.log(`WebGPU + q4:  ${s.webgpu_q4_works ? 'WORKS' : 'failed'}${s.webgpu_q4_infer_ms ? ` (~${s.webgpu_q4_infer_ms}ms infer)` : ''}`);
  console.log(`WebGPU + q4f16: ${s.q4f16_any_ok ? 'WORKS' : 'blocked — needs shader-f16 GPU feature'}`);
  console.log(`shader-f16 seen: ${s.q4f16_shader_f16_available}`);
  console.log(`${s.succeeded}/${s.total} strategies succeeded`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

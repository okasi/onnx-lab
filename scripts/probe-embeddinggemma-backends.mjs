#!/usr/bin/env node
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const worker = path.join(__dirname, 'probe-embeddinggemma-worker.mjs');

const STRATEGIES = ['wasm-asyncify', 'wasm-jsep', 'webgpu', 'cpu'];

async function main() {
  console.log('EmbeddingGemma 300M q4f16 backend probe\n');
  const results = [];

  for (const strategy of STRATEGIES) {
    process.stdout.write(`→ ${strategy} … `);
    const r = await runWorker(strategy);
    results.push(r);
    if (r.status === 'ok') {
      console.log(`ok dim=${r.dim} infer=${r.ms}ms load=${r.load_ms}ms`);
    } else {
      console.log(`fail: ${r.error}`);
    }
  }

  const outPath = path.join(root, 'results', `probe-embeddinggemma-q4f16-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify({ results }, null, 2));
  console.log(`\nWrote ${outPath}`);
  console.log(`${results.filter((r) => r.status === 'ok').length}/${results.length} node probes succeeded`);
}

function runWorker(strategy) {
  return new Promise((resolve) => {
    const child = spawn(process.execPath, ['--expose-gc', worker, strategy], {
      cwd: root,
      env: { ...process.env },
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
        resolve({ strategy, status: 'error', error: stderr.trim() || stdout.trim() || 'parse error' });
      }
    });
  });
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

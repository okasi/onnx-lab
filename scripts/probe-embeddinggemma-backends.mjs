#!/usr/bin/env node
import path from 'node:path';
import {
  RESULTS_DIR,
  SCRIPTS_DIR,
  runJsonWorker,
  writeJson,
} from '../lib/benchmark-support.mjs';

const workerScript = path.join(SCRIPTS_DIR, 'probe-embeddinggemma-worker.mjs');
const STRATEGIES = ['wasm-asyncify', 'wasm-jsep', 'cpu'];

async function main() {
  console.log('EmbeddingGemma 300M q4f16 backend probe\n');
  const results = [];
  for (const strategy of STRATEGIES) {
    process.stdout.write(`${strategy} ... `);
    const result = (await runJsonWorker(
      workerScript,
      [strategy],
      { resultFile: false },
    )).result;
    results.push(result);
    console.log(
      result.status === 'ok'
        ? `ok dim=${result.dim} infer=${result.ms}ms load=${result.load_ms}ms`
        : `failed: ${result.error}`,
    );
  }

  const outPath = path.join(
    RESULTS_DIR,
    `probe-embeddinggemma-q4f16-${Date.now()}.json`,
  );
  await writeJson(outPath, { results });
  console.log(`\nWrote ${outPath}`);
  console.log(`${results.filter((result) => result.status === 'ok').length}/${results.length} probes succeeded`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

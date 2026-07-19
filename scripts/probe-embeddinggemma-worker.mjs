#!/usr/bin/env node
import { createFeatureExtractor } from '../lib/transformers-runtime.mjs';
import { dispose } from '../lib/benchmark-support.mjs';

const strategy = process.argv[2];
const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const DTYPE = 'q4f16';
const TEST_TEXT = 'Stockholm är huvudstaden i Sverige.';

const BACKEND_MAP = {
  'wasm-asyncify': 'wasm',
  'wasm-jsep': 'wasm-jsep',
  cpu: 'cpu',
};

async function run() {
  const result = { strategy, status: 'pending', error: null };
  const backend = BACKEND_MAP[strategy];
  if (!backend) {
    result.status = 'error';
    result.error = `unknown strategy: ${strategy}`;
    console.log(JSON.stringify(result));
    return;
  }

  try {
    const loadStart = performance.now();
    const extractor = await createFeatureExtractor(
      MODEL_ID,
      { dtype: DTYPE },
      backend,
    );
    result.load_ms = Math.round(performance.now() - loadStart);
    result.backend_used = extractor._benchmark_backend ?? backend;

    const inferStart = performance.now();
    const out = await extractor(TEST_TEXT, { pooling: 'mean', normalize: true });
    result.ms = Math.round(performance.now() - inferStart);
    const data = Array.from(out.data ?? out.tolist?.()[0] ?? []);
    result.dim = data.length;
    result.sample = data.slice(0, 3).map((x) => Number(x.toFixed(6)));
    result.status = 'ok';
    await dispose(extractor);
  } catch (e) {
    result.status = 'error';
    result.error = (e?.message ?? String(e)).slice(0, 600);
  }

  console.log(JSON.stringify(result));
}

run();

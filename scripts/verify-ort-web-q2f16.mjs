#!/usr/bin/env node
/**
 * Verify local onnxruntime-web loads Gemma 4 mobile embed_tokens via WASM JSEP.
 */
import path from 'node:path';
import fs from 'node:fs';
import { projectPath, readJson } from '../lib/benchmark-support.mjs';
import { bootstrapOrt } from '../lib/transformers-runtime.mjs';

const embedPath = projectPath(
  '.cache/transformers-node/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX/onnx/embed_tokens_q2f16.onnx',
);

async function main() {
  if (!fs.existsSync(embedPath)) {
    throw new Error(`Missing cached model file: ${embedPath}`);
  }
  const pkg = await readJson(projectPath('node_modules', 'onnxruntime-web', 'package.json'));
  console.log(`onnxruntime-web version: ${pkg.version}`);

  const dataPath = embedPath.replace(/\.onnx$/, '.onnx_data');
  const externalData = fs.existsSync(dataPath)
    ? [{ path: path.basename(dataPath), data: fs.readFileSync(dataPath) }]
    : [];

  const ort = await bootstrapOrt({ wasmVariant: 'jsep' });
  const session = await ort.InferenceSession.create(embedPath, {
    executionProviders: ['wasm'],
    externalData,
  });
  console.log('embed_tokens_q2f16 (wasm-jsep): OK');
  console.log('  inputs:', session.inputNames);
  console.log('  outputs:', session.outputNames);
}

main().catch((e) => {
  console.error('verify failed:', e.message ?? e);
  process.exit(1);
});

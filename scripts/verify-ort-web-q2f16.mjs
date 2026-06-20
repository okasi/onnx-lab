#!/usr/bin/env node
/**
 * Verify local onnxruntime-web loads Gemma 4 mobile embed_tokens via WASM JSEP.
 */
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { bootstrapOrt } from '../lib/transformers-runtime.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const embedPath = path.join(
  root,
  '.cache/transformers-node/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX/onnx/embed_tokens_q2f16.onnx',
);

async function main() {
  const pkg = JSON.parse(
    fs.readFileSync(path.join(root, 'node_modules/onnxruntime-web/package.json'), 'utf8'),
  );
  console.log(`onnxruntime-web version: ${pkg.version}`);

  const dataPath = embedPath.replace(/\.onnx$/, '.onnx_data');
  const externalData = fs.existsSync(dataPath)
    ? [{ path: path.basename(dataPath), data: fs.readFileSync(dataPath) }]
    : [];

  const ort = await bootstrapOrt({ bundle: 'wasm', wasmVariant: 'jsep' });
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

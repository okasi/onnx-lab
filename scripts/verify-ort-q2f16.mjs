#!/usr/bin/env node
/**
 * Smoke test: local ORT 1.27+ loads Gemma 4 mobile embed_tokens (2-bit GatherBlockQuantized).
 */
import fs from 'node:fs/promises';
import { projectPath } from '../lib/benchmark-support.mjs';

const embedPath = projectPath(
  '.cache/transformers-node/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX/onnx/embed_tokens_q2f16.onnx',
);

async function main() {
  await fs.access(embedPath).catch(() => {
    throw new Error(`Missing cached model file: ${embedPath}`);
  });
  let ort;
  try {
    ort = await import('onnxruntime-node');
  } catch (e) {
    console.error('onnxruntime-node not installed. Run: npm install (ORT 1.27.0) — see docs/ort-127-install.md');
    throw e;
  }

  const ver = (await import('onnxruntime-node/package.json', { with: { type: 'json' } })).default.version;
  console.log(`onnxruntime-node version: ${ver}`);

  const session = await ort.InferenceSession.create(embedPath, { executionProviders: ['cpu'] });
  console.log('embed_tokens_q2f16: OK');
  console.log('  inputs:', session.inputNames);
  console.log('  outputs:', session.outputNames);
}

main().catch((e) => {
  console.error('verify failed:', e.message ?? e);
  process.exit(1);
});

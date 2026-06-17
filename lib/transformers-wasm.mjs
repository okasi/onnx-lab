import * as ort from 'onnxruntime-web';
import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const dist = path.join(root, 'node_modules/onnxruntime-web/dist');

const wasmMjs = path.join(dist, 'ort-wasm-simd-threaded.asyncify.mjs');
const wasmBin = path.join(dist, 'ort-wasm-simd-threaded.asyncify.wasm');

ort.env.wasm.wasmPaths = {
  mjs: pathToFileURL(wasmMjs).href,
  wasm: pathToFileURL(wasmBin).href,
};
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmBinary = await fs.readFile(wasmBin);
ort.env.wasm.proxy = false;

globalThis[Symbol.for('onnxruntime')] = ort;

const tfm = await import('@huggingface/transformers');

tfm.env.allowRemoteModels = true;
tfm.env.useFSCache = true;
tfm.env.cacheDir = path.join(root, '.cache', 'transformers-node');

export const { env, ModelRegistry } = tfm;

export async function createFeatureExtractor(modelId, options = {}) {
  return tfm.pipeline('feature-extraction', modelId, {
    ...options,
    device: 'auto',
    session_options: {
      executionProviders: ['wasm'],
      ...options.session_options,
    },
  });
}

import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

async function loadTransformers() {
  const tfm = await import('@huggingface/transformers');
  tfm.env.allowRemoteModels = true;
  tfm.env.useFSCache = true;
  tfm.env.cacheDir = path.join(root, '.cache', 'transformers-node');
  return tfm;
}

async function bootstrapWasm() {
  const ort = await import('onnxruntime-web');
  const dist = path.join(root, 'node_modules/onnxruntime-web/dist');
  ort.env.wasm.wasmPaths = {
    mjs: pathToFileURL(path.join(dist, 'ort-wasm-simd-threaded.asyncify.mjs')).href,
    wasm: pathToFileURL(path.join(dist, 'ort-wasm-simd-threaded.asyncify.wasm')).href,
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmBinary = await fs.readFile(
    path.join(dist, 'ort-wasm-simd-threaded.asyncify.wasm'),
  );
  ort.env.wasm.proxy = false;
  globalThis[Symbol.for('onnxruntime')] = ort;
}

export function isWasmIncompatibleError(message) {
  const m = message.toLowerCase();
  return (
    m.includes('gatherblockquantized') ||
    m.includes('onnx_data') ||
    m.includes('mountedfiles') ||
    m.includes('external data') ||
    m.includes('bad_alloc') ||
    m.includes('std::bad_alloc')
  );
}

/** @param {'wasm'|'cpu'} backend */
export async function createFeatureExtractor(modelId, options = {}, backend = 'cpu') {
  if (backend === 'wasm') {
    await bootstrapWasm();
    const tfm = await loadTransformers();
    const extractor = await tfm.pipeline('feature-extraction', modelId, {
      ...options,
      device: 'auto',
      session_options: {
        executionProviders: ['wasm'],
        ...options.session_options,
      },
    });
    extractor._benchmark_backend = 'wasm';
    return extractor;
  }

  if (globalThis[Symbol.for('onnxruntime')]) {
    delete globalThis[Symbol.for('onnxruntime')];
  }

  const tfm = await loadTransformers();
  const extractor = await tfm.pipeline('feature-extraction', modelId, {
    ...options,
    device: 'cpu',
  });
  extractor._benchmark_backend = 'cpu';
  return extractor;
}

export async function getModelRegistry() {
  const tfm = await loadTransformers();
  return tfm.ModelRegistry;
}

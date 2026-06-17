import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';

/** @typedef {'asyncify'|'jsep'} WasmVariant */
/** @typedef {'wasm'|'jsep'|'webgpu'} OrtBundle */

/**
 * @param {object} opts
 * @param {OrtBundle} [opts.bundle]
 * @param {WasmVariant} [opts.wasmVariant]
 */
export async function bootstrapOrt({ bundle = 'wasm', wasmVariant = 'asyncify' } = {}) {
  const dist = path.join(root, 'node_modules/onnxruntime-web/dist');
  const ort =
    bundle === 'webgpu'
      ? await import('onnxruntime-web/webgpu')
      : await import('onnxruntime-web');

  const files = {
    asyncify: ['ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm'],
    jsep: ['ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm'],
  };
  const [mjsName, wasmName] = files[wasmVariant] ?? files.asyncify;

  ort.env.wasm.wasmPaths = {
    mjs: pathToFileURL(path.join(dist, mjsName)).href,
    wasm: pathToFileURL(path.join(dist, wasmName)).href,
  };
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmBinary = await fs.readFile(path.join(dist, wasmName));

  if (globalThis[Symbol.for('onnxruntime')]) {
    delete globalThis[Symbol.for('onnxruntime')];
  }
  globalThis[Symbol.for('onnxruntime')] = ort;
  return ort;
}

/** Dawn WebGPU polyfill for Node (needs Vulkan — fails on headless VMs without lavapipe). */
export async function initWebGpuPolyfill() {
  const { create, globals } = await import('webgpu');
  Object.assign(globalThis, globals);
  const instance = create([]);
  const nav = globalThis.navigator ?? {};
  nav.gpu = instance;
  if (!globalThis.navigator) {
    Object.defineProperty(globalThis, 'navigator', {
      value: nav,
      writable: true,
      configurable: true,
    });
  }
  return instance;
}

async function loadTransformers() {
  const tfm = await import('@huggingface/transformers');
  tfm.env.allowRemoteModels = true;
  tfm.env.useFSCache = true;
  tfm.env.useWasmCache = false;
  tfm.env.cacheDir = path.join(root, '.cache', 'transformers-node');
  return tfm;
}

/**
 * Build externalData session options for ONNX shards Transformers.js omits in Node.
 * @param {string} modelId
 * @param {string} dtype
 * @param {string} [modelFileName]
 */
export async function externalDataSessionOptions(modelId, dtype, modelFileName = 'model') {
  const suffixMap = {
    bnb4: '_bnb4',
    int8: '_int8',
    q4: '_q4',
    q4f16: '_q4f16',
    q8: '_quantized',
    uint8: '_uint8',
    fp32: '',
  };
  const suffix = suffixMap[dtype] ?? `_${dtype}`;
  const base = `${modelFileName}${suffix}`;
  const rel = `${base}.onnx_data`;
  const hubDir = modelId.replace(/\//g, path.sep);
  const abs = path.join(root, '.cache', 'transformers-node', hubDir, 'onnx', rel);
  try {
    await fs.access(abs);
    return {
      externalData: [{ path: rel, data: abs }],
    };
  } catch {
    return {};
  }
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

/**
 * @param {'wasm'|'wasm-jsep'|'webgpu'|'cpu'} backend
 * @param {string} modelId
 * @param {object} options
 */
export async function createFeatureExtractor(modelId, options = {}, backend = 'cpu') {
  const dtype = options.dtype ?? 'fp32';
  const modelFileName = options.model_file_name ?? 'model';
  const extOpts = await externalDataSessionOptions(modelId, dtype, modelFileName);

  if (backend === 'wasm' || backend === 'wasm-jsep') {
    await bootstrapOrt({
      bundle: 'wasm',
      wasmVariant: backend === 'wasm-jsep' ? 'jsep' : 'asyncify',
    });
    const tfm = await loadTransformers();
    const extractor = await tfm.pipeline('feature-extraction', modelId, {
      ...options,
      device: 'auto',
      session_options: {
        executionProviders: ['wasm'],
        ...extOpts,
        ...options.session_options,
      },
    });
    extractor._benchmark_backend = backend;
    return extractor;
  }

  if (backend === 'webgpu') {
    try {
      await initWebGpuPolyfill();
    } catch {
      // polyfill optional — browser has native WebGPU
    }
    await bootstrapOrt({ bundle: 'webgpu', wasmVariant: 'jsep' });
    const tfm = await loadTransformers();
    const extractor = await tfm.pipeline('feature-extraction', modelId, {
      ...options,
      device: 'auto',
      session_options: {
        executionProviders: ['webgpu'],
        enableMemPattern: false,
        ...extOpts,
        ...options.session_options,
      },
    });
    extractor._benchmark_backend = 'webgpu';
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

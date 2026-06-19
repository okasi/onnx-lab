import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fileURLToPath } from 'node:url';
import { gemma4Suffix } from '../config/gemma4-models.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

const DTYPE_SUFFIX_MAP = {
  bnb4: '_bnb4',
  int8: '_int8',
  q4: '_q4',
  q4f16: '_q4f16',
  q8: '_quantized',
  uint8: '_uint8',
  fp32: '',
  fp16: '_fp16',
  q2f16: '_q2f16',
};

const GEMMA4_TEXT_SESSIONS = ['embed_tokens', 'decoder_model_merged'];

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
  const suffix = DTYPE_SUFFIX_MAP[dtype] ?? `_${dtype}`;
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

function externalDataChunkNames(baseName, numChunks) {
  const names = [];
  for (let i = 0; i < numChunks; i += 1) {
    names.push(`${baseName}.onnx_data${i === 0 ? '' : `_${i}`}`);
  }
  return names;
}

function resolveExternalDataChunkCount(extFmt, onnxFileName, sessionFileName) {
  if (!extFmt) {
    return 0;
  }
  if (typeof extFmt === 'number') {
    return extFmt;
  }
  if (extFmt[onnxFileName] != null) {
    return Number(extFmt[onnxFileName]);
  }
  if (extFmt[sessionFileName] != null) {
    return Number(extFmt[sessionFileName]);
  }
  return 0;
}

/**
 * Hub-relative externalData entries for Gemma 4 text sessions (WASM in Node).
 * @param {object} transformersJsConfig
 * @param {string} dtype
 */
export function gemma4WasmExternalData(transformersJsConfig, dtype) {
  const extFmt = transformersJsConfig?.use_external_data_format ?? {};
  const suffix = gemma4Suffix(dtype);
  const externalData = [];
  for (const fileName of GEMMA4_TEXT_SESSIONS) {
    const base = `${fileName}${suffix}`;
    const onnxName = `${base}.onnx`;
    const numChunks = resolveExternalDataChunkCount(extFmt, onnxName, fileName) || 1;
    for (const dataName of externalDataChunkNames(base, numChunks)) {
      externalData.push({ path: dataName, data: `onnx/${dataName}` });
    }
  }
  return externalData;
}

/**
 * @param {'wasm'|'wasm-jsep'|'webgpu'|'cpu'} backend
 * @param {string} modelId
 * @param {object} options
 */
export async function createTextGenerator(modelId, options = {}, backend = 'cpu') {
  const dtype = options.dtype ?? 'fp32';

  if (backend === 'wasm' || backend === 'wasm-jsep') {
    await bootstrapOrt({
      bundle: 'wasm',
      wasmVariant: backend === 'wasm-jsep' ? 'jsep' : 'asyncify',
    });
  } else if (backend === 'webgpu') {
    try {
      await initWebGpuPolyfill();
    } catch {
      // polyfill optional — browser has native WebGPU
    }
    await bootstrapOrt({ bundle: 'webgpu', wasmVariant: 'jsep' });
  } else if (globalThis[Symbol.for('onnxruntime')]) {
    delete globalThis[Symbol.for('onnxruntime')];
  }

  const tfm = await loadTransformers();
  const config = await tfm.AutoConfig.from_pretrained(modelId, {
    cache_dir: tfm.env.cacheDir,
  });
  const tjsConfig = config['transformers.js_config'] ?? {};

  const pipelineOptions = {
    ...options,
    dtype,
  };

  if (backend === 'wasm' || backend === 'wasm-jsep') {
    pipelineOptions.device = 'auto';
    pipelineOptions.use_external_data_format = {};
    pipelineOptions.session_options = {
      executionProviders: ['wasm'],
      externalData: gemma4WasmExternalData(tjsConfig, dtype),
      ...options.session_options,
    };
    const generator = await tfm.pipeline('text-generation', modelId, pipelineOptions);
    generator._benchmark_backend = backend;
    return generator;
  }

  if (backend === 'webgpu') {
    const webgpuDtype = options.webgpu_dtype ?? dtype;
    const sessionOpts = {
      executionProviders: ['webgpu'],
      enableMemPattern: false,
      ...options.session_options,
    };
    try {
      const generator = await tfm.pipeline('text-generation', modelId, {
        ...pipelineOptions,
        dtype: webgpuDtype,
        device: 'auto',
        session_options: sessionOpts,
      });
      generator._benchmark_backend = 'webgpu';
      return generator;
    } catch (e) {
      const msg = e?.message ?? String(e);
      const needsF16 = msg.includes('requires f16') || msg.includes('shader-f16');
      const fallback = options.webgpu_fallback_dtype ?? 'q4';
      if (needsF16 && webgpuDtype !== fallback && fallback) {
        const generator = await tfm.pipeline('text-generation', modelId, {
          ...pipelineOptions,
          dtype: fallback,
          device: 'auto',
          session_options: sessionOpts,
        });
        generator._benchmark_backend = 'webgpu';
        generator._webgpu_dtype_fallback = { from: webgpuDtype, to: fallback };
        return generator;
      }
      throw e;
    }
  }

  const generator = await tfm.pipeline('text-generation', modelId, {
    ...pipelineOptions,
    device: 'cpu',
  });
  generator._benchmark_backend = 'cpu';
  return generator;
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
    const webgpuDtype = options.webgpu_dtype ?? dtype;
    const sessionOpts = {
      executionProviders: ['webgpu'],
      enableMemPattern: false,
      ...extOpts,
      ...options.session_options,
    };
    try {
      const extractor = await tfm.pipeline('feature-extraction', modelId, {
        ...options,
        dtype: webgpuDtype,
        device: 'auto',
        session_options: sessionOpts,
      });
      extractor._benchmark_backend = 'webgpu';
      return extractor;
    } catch (e) {
      const msg = e?.message ?? String(e);
      const needsF16 = msg.includes('requires f16') || msg.includes('shader-f16');
      const fallback = options.webgpu_fallback_dtype ?? 'q4';
      if (needsF16 && webgpuDtype === 'q4f16' && fallback && fallback !== webgpuDtype) {
        const fallbackExt = await externalDataSessionOptions(modelId, fallback, modelFileName);
        const extractor = await tfm.pipeline('feature-extraction', modelId, {
          ...options,
          dtype: fallback,
          device: 'auto',
          session_options: { ...sessionOpts, ...fallbackExt },
        });
        extractor._benchmark_backend = 'webgpu';
        extractor._webgpu_dtype_fallback = { from: webgpuDtype, to: fallback };
        return extractor;
      }
      throw e;
    }
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

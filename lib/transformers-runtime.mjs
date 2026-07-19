import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { dtypeSuffix } from '../config/models.mjs';
import { ROOT_DIR } from './benchmark-support.mjs';

const ORT_SYMBOL = Symbol.for('onnxruntime');
const GEMMA4_TEXT_SESSIONS = ['embed_tokens', 'decoder_model_merged'];

/** @typedef {'asyncify'|'jsep'} WasmVariant */

/**
 * @param {object} opts
 * @param {WasmVariant} [opts.wasmVariant]
 */
export async function bootstrapOrt({ wasmVariant = 'asyncify' } = {}) {
  const dist = path.join(ROOT_DIR, 'node_modules/onnxruntime-web/dist');
  const ort = await import('onnxruntime-web');

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

  resetOrt();
  globalThis[ORT_SYMBOL] = ort;
  return ort;
}

function resetOrt() {
  if (globalThis[ORT_SYMBOL]) {
    delete globalThis[ORT_SYMBOL];
  }
}

async function loadTransformers() {
  const tfm = await import('@huggingface/transformers');
  tfm.env.allowRemoteModels = true;
  tfm.env.useFSCache = true;
  tfm.env.useWasmCache = false;
  tfm.env.cacheDir = path.join(ROOT_DIR, '.cache', 'transformers-node');
  return tfm;
}

async function initializeBackend(backend) {
  if (backend === 'wasm' || backend === 'wasm-jsep') {
    await bootstrapOrt({
      wasmVariant: backend === 'wasm-jsep' ? 'jsep' : 'asyncify',
    });
  } else if (backend === 'cpu') {
    resetOrt();
  } else {
    throw new Error(`Unknown backend: ${backend}`);
  }
  return loadTransformers();
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

function externalDataEntries(transformersJsConfig, dtype, fileNames) {
  const extFmt = transformersJsConfig?.use_external_data_format;
  if (!extFmt) {
    return [];
  }
  const suffix = dtypeSuffix(dtype);
  const entries = [];
  for (const fileName of fileNames) {
    const base = `${fileName}${suffix}`;
    const onnxName = `${base}.onnx`;
    const numChunks = resolveExternalDataChunkCount(extFmt, onnxName, fileName) || 1;
    for (const dataName of externalDataChunkNames(base, numChunks)) {
      entries.push({ path: dataName, data: `onnx/${dataName}` });
    }
  }
  return entries;
}

export function externalDataSessionOptions(
  transformersJsConfig,
  dtype,
  modelFileName = 'model',
) {
  const externalData = externalDataEntries(
    transformersJsConfig,
    dtype,
    [modelFileName],
  );
  return externalData.length ? { externalData } : {};
}

/**
 * Hub-relative externalData entries for Gemma 4 text sessions (WASM in Node).
 * @param {object} transformersJsConfig
 * @param {string} dtype
 */
export function gemma4WasmExternalData(transformersJsConfig, dtype) {
  return externalDataEntries(transformersJsConfig, dtype, GEMMA4_TEXT_SESSIONS);
}

/**
 * @param {'wasm'|'wasm-jsep'|'cpu'} backend
 * @param {string} modelId
 * @param {object} options
 */
export async function createTextGenerator(modelId, options = {}, backend = 'cpu') {
  const dtype = options.dtype ?? 'fp32';
  const tfm = await initializeBackend(backend);
  let tjsConfig = {};
  if (backend !== 'cpu') {
    const config = await tfm.AutoConfig.from_pretrained(modelId, {
      cache_dir: tfm.env.cacheDir,
    });
    tjsConfig = config['transformers.js_config'] ?? {};
  }

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

  const generator = await tfm.pipeline('text-generation', modelId, {
    ...pipelineOptions,
    device: 'cpu',
  });
  generator._benchmark_backend = 'cpu';
  return generator;
}

/**
 * @param {'wasm'|'wasm-jsep'|'cpu'} backend
 * @param {string} modelId
 * @param {object} options
 */
export async function createFeatureExtractor(modelId, options = {}, backend = 'cpu') {
  const dtype = options.dtype ?? 'fp32';
  const modelFileName = options.model_file_name ?? 'model';
  const tfm = await initializeBackend(backend);
  let transformersJsConfig = {};
  if (backend !== 'cpu') {
    const config = await tfm.AutoConfig.from_pretrained(modelId, {
      cache_dir: tfm.env.cacheDir,
    });
    transformersJsConfig = config['transformers.js_config'] ?? {};
  }
  const extOpts = externalDataSessionOptions(
    transformersJsConfig,
    dtype,
    modelFileName,
  );

  if (backend === 'wasm' || backend === 'wasm-jsep') {
    const extractor = await tfm.pipeline('feature-extraction', modelId, {
      ...options,
      device: 'auto',
      use_external_data_format: {},
      session_options: {
        executionProviders: ['wasm'],
        ...extOpts,
        ...options.session_options,
      },
    });
    extractor._benchmark_backend = backend;
    return extractor;
  }

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

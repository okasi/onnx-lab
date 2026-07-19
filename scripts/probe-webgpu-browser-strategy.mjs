#!/usr/bin/env node
import path from 'node:path';
import { ROOT_DIR } from '../lib/benchmark-support.mjs';
import { runBrowserHtml } from '../lib/browser-runtime.mjs';

const strategy = process.argv[2];
const MODEL_ID = 'onnx-community/embeddinggemma-300m-ONNX';
const TEST_TEXT = 'Stockholm är huvudstaden i Sverige.';

const SPECS = {
  'browser:default': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:vulkan-angle': {
    args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:swiftshader': {
    args: ['--enable-unsafe-webgpu', '--use-angle=swiftshader-webgl'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:headed': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: false,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:force-cpu-gather': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: JSON.stringify({
      executionProviders: [{
        name: 'webgpu',
        forceCpuNodeNames: ['/model/embed_tokens/Gather_Q4'],
      }],
    }),
  },
  'browser:webgpu-wasm-dual': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: JSON.stringify({ executionProviders: ['webgpu', 'wasm'] }),
  },
  'browser:q4-control': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4',
    sessionJson: '{}',
  },
  'browser:highperf-adapter': {
    args: [
      '--enable-unsafe-webgpu',
      '--use-angle=default',
      '--enable-features=WebGPUService',
    ],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
    adapterOpts: '{ powerPreference: "high-performance" }',
  },
  'browser:angle-gl': {
    args: ['--enable-unsafe-webgpu', '--use-angle=gl', '--ignore-gpu-blocklist'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:chrome-blog-flags': {
    args: [
      '--headless=new',
      '--no-sandbox',
      '--use-angle=vulkan',
      '--enable-features=Vulkan',
      '--disable-vulkan-surface',
      '--enable-unsafe-webgpu',
    ],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
    env: {},
  },
  'browser:lavapipe-vk': {
    args: [
      '--headless=new',
      '--no-sandbox',
      '--use-angle=vulkan',
      '--enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE',
      '--disable-vulkan-surface',
      '--enable-unsafe-webgpu',
      '--enable-features=WebGPUDeveloperFeatures',
    ],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
    env: 'lavapipe',
  },
};

function htmlFor(spec) {
  const sessionOpts = spec.sessionJson || '{}';
  const adapterOpts = spec.adapterOpts || '{ powerPreference: "high-performance" }';
  return `<!DOCTYPE html><html><body><pre id="log">…</pre><script type="module">
const log = (m) => { document.getElementById('log').textContent += '\\n' + m; };
try {
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  if (!navigator.gpu) throw new Error('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter(${adapterOpts});
  if (!adapter) throw new Error('no adapter');
  const info = adapter.info ?? {};
  const shaderF16 = adapter.features.has('shader-f16');
  log('adapter=' + (info.description || info.vendor || JSON.stringify(info)));
  log('shader-f16=' + shaderF16);
  log('features=' + [...adapter.features].sort().join(','));
  const req = shaderF16 ? ['shader-f16'] : [];
  const device = await adapter.requestDevice({ requiredFeatures: req });
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  env.backends.onnx.webgpu.device = device;
  const sessionOpts = Object.assign({ enableMemPattern: false }, ${sessionOpts});
  const t0 = performance.now();
  const extractor = await pipeline('feature-extraction', '${MODEL_ID}', {
    dtype: '${spec.dtype}',
    device: 'webgpu',
    session_options: sessionOpts,
  });
  const loadMs = Math.round(performance.now() - t0);
  log('load ' + loadMs + 'ms');
  const t1 = performance.now();
  const out = await extractor('${TEST_TEXT}', { pooling: 'mean', normalize: true });
  const inferMs = Math.round(performance.now() - t1);
  const data = out.tolist ? out.tolist()[0] : Array.from(out.data);
  log('infer ' + inferMs + 'ms dim=' + data.length);
  window.__RESULT__ = {
    status: 'ok', dim: data.length, load_ms: loadMs, infer_ms: inferMs,
    shader_f16: shaderF16,
    sample: data.slice(0,3).map(x=>+Number(x).toFixed(6)),
  };
} catch (e) {
  log('ERROR: ' + e.message);
  window.__RESULT__ = { status: 'error', error: e.message };
}
</script></body></html>`;
}

async function main() {
  const spec = SPECS[strategy];
  const result = { strategy, status: 'pending', error: null };
  if (!spec) {
    result.status = 'error';
    result.error = `unknown browser strategy: ${strategy}`;
    console.log(JSON.stringify(result));
    return;
  }

  const launchEnv = { ...process.env };
  if (spec.env === 'lavapipe') {
    const base = path.join(ROOT_DIR, '.cache/vulkan/extracted');
    const icd = path.join(ROOT_DIR, '.cache/vulkan/lvp_icd_abs.json');
    const lib = path.join(base, 'usr/lib/x86_64-linux-gnu');
    launchEnv.VK_DRIVER_FILES = icd;
    launchEnv.VK_ICD_FILENAMES = icd;
    launchEnv.LD_LIBRARY_PATH = `${lib}:${launchEnv.LD_LIBRARY_PATH ?? ''}`;
  }

  try {
    const { payload } = await runBrowserHtml(htmlFor(spec), {
      timeoutMs: 600_000,
      logElementId: 'log',
      launchOptions: {
        headless: spec.headless,
        args: spec.args,
        env: launchEnv,
      },
    });
    Object.assign(result, payload);
  } catch (e) {
    result.status = 'error';
    result.error = (e?.message ?? String(e)).slice(0, 500);
  }
  console.log(JSON.stringify(result));
}

main();

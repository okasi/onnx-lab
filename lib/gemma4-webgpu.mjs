import { runBrowserHtml } from './browser-runtime.mjs';
import { round } from './benchmark-support.mjs';

/** WebGPU browser strategies for Gemma 4 text-generation probes. */
export const GEMMA4_WEBGPU_STRATEGIES = {
  'browser:q4-control': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4',
    sessionJson: '{}',
  },
  'browser:q4-force-gather': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4',
    sessionJson: JSON.stringify({
      executionProviders: [{
        name: 'webgpu',
        forceCpuNodeNames: [
          '/model/embed_tokens/Gather_Quant',
          '/model/embed_tokens/Gather_Q4',
        ],
      }],
    }),
  },
  'browser:q4-dual-ep': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4',
    sessionJson: JSON.stringify({ executionProviders: ['webgpu', 'wasm'] }),
  },
  'browser:q4f16': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q4f16',
    sessionJson: '{}',
  },
  'browser:q8': {
    args: ['--enable-unsafe-webgpu', '--use-angle=default'],
    headless: true,
    dtype: 'q8',
    sessionJson: '{}',
  },
  'browser:angle-gl': {
    args: ['--enable-unsafe-webgpu', '--use-angle=gl', '--ignore-gpu-blocklist'],
    headless: true,
    dtype: 'q4',
    sessionJson: '{}',
  },
  'browser:vulkan-angle': {
    args: ['--enable-unsafe-webgpu', '--use-angle=vulkan', '--enable-features=Vulkan'],
    headless: true,
    dtype: 'q4',
    sessionJson: '{}',
  },
};

export function gemma4BrowserHtml(modelId, spec, prompts, maxNewTokens) {
  const sessionOpts = spec.sessionJson || '{}';
  const promptsJson = JSON.stringify(prompts);
  return `<!DOCTYPE html><html><body><script type="module">
const prompts = ${promptsJson};
try {
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  if (!navigator.gpu) throw new Error('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const shaderF16 = adapter.features.has('shader-f16');
  const device = await adapter.requestDevice({ requiredFeatures: shaderF16 ? ['shader-f16'] : [] });
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  env.backends.onnx.webgpu.device = device;
  let usedDtype = '${spec.dtype}';
  const sessionOpts = Object.assign({ enableMemPattern: false }, ${sessionOpts});
  const loadStart = performance.now();
  let generator;
  try {
    generator = await pipeline('text-generation', '${modelId}', { dtype: usedDtype, device: 'webgpu', session_options: sessionOpts });
  } catch (e) {
    const msg = e?.message ?? String(e);
    if ((msg.includes('shader-f16') || msg.includes('requires f16')) && usedDtype !== 'q4') {
      usedDtype = 'q4';
      generator = await pipeline('text-generation', '${modelId}', { dtype: usedDtype, device: 'webgpu', session_options: sessionOpts });
    } else {
      throw e;
    }
  }
  const loadMs = performance.now() - loadStart;
  const latencies = [];
  const outputs = [];
  for (const item of prompts) {
    const inferStart = performance.now();
    const out = await generator(item.prompt, { max_new_tokens: ${maxNewTokens}, do_sample: false });
    latencies.push(performance.now() - inferStart);
    outputs.push({ ...item, generated_text: out?.[0]?.generated_text ?? null });
  }
  const sorted = [...latencies].sort((a, b) => a - b);
  const totalMs = latencies.reduce((sum, value) => sum + value, 0);
  const percentile = (ratio) => Math.round(sorted[Math.min(sorted.length - 1, Math.floor(ratio * (sorted.length - 1)))]);
  window.__RESULT__ = {
    status: 'ok',
    load_ms: Math.round(loadMs),
    inference: {
      count: latencies.length,
      mean_ms: Math.round(totalMs / latencies.length),
      total_ms: Math.round(totalMs),
      p50_ms: percentile(0.5),
      p95_ms: percentile(0.95),
    },
    outputs,
    shader_f16: shaderF16,
    dtype_used: usedDtype,
  };
} catch (e) {
  window.__RESULT__ = { status: 'error', error: e.message };
}
</script></body></html>`;
}

export async function runGemma4BrowserStrategy({
  modelId,
  strategyName,
  prompts,
  maxNewTokens,
  dtype = null,
  timeoutMs = 600_000,
}) {
  const spec = GEMMA4_WEBGPU_STRATEGIES[strategyName];
  if (!spec) {
    throw new Error(`Unknown WebGPU strategy: ${strategyName}`);
  }
  const effectiveSpec = dtype ? { ...spec, dtype } : spec;
  const { payload, total_ms } = await runBrowserHtml(
    gemma4BrowserHtml(modelId, effectiveSpec, prompts, maxNewTokens),
    {
      timeoutMs,
      launchOptions: {
        headless: effectiveSpec.headless ?? true,
        args: effectiveSpec.args,
      },
    },
  );
  if (payload.status !== 'ok') {
    throw new Error(payload.error ?? 'WebGPU browser run failed');
  }
  return {
    ...payload,
    total_ms: round(total_ms),
    strategy: strategyName,
  };
}

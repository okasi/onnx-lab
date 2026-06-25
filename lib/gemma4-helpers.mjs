/**
 * Shared helpers for Gemma 4 probes and benchmarks.
 */

export function round(n, digits = 2) {
  return Number(Number(n).toFixed(digits));
}

export function snapshotMemory() {
  const m = process.memoryUsage();
  return {
    rss_mb: round(m.rss / 1024 / 1024),
    heap_used_mb: round(m.heapUsed / 1024 / 1024),
    external_mb: round(m.external / 1024 / 1024),
  };
}

export class MemoryMonitor {
  constructor() {
    this.peak = { rss_mb: 0, heap_used_mb: 0, external_mb: 0 };
    this.interval = null;
  }

  start() {
    this.sample();
    this.interval = setInterval(() => this.sample(), 200);
  }

  sample() {
    const s = snapshotMemory();
    for (const key of Object.keys(this.peak)) {
      if (s[key] > this.peak[key]) this.peak[key] = s[key];
    }
  }

  stop() {
    if (this.interval) clearInterval(this.interval);
    this.sample();
    return {
      peak_rss_mb: this.peak.rss_mb,
      peak_heap_used_mb: this.peak.heap_used_mb,
      peak_external_mb: this.peak.external_mb,
    };
  }
}

export function summarizeTimings(values) {
  if (!values.length) {
    return { count: 0, mean_ms: 0, total_ms: 0, p50_ms: 0, p95_ms: 0 };
  }
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((a, b) => a + b, 0);
  const p = (q) => sorted[Math.min(sorted.length - 1, Math.floor(q * (sorted.length - 1)))];
  return {
    count: values.length,
    mean_ms: round(sum / values.length),
    total_ms: round(sum),
    p50_ms: round(p(0.5)),
    p95_ms: round(p(0.95)),
  };
}

export function classifyGemma4Error(message) {
  const m = message.toLowerCase();
  if (m.includes('bad_alloc') || m.includes('out of memory') || m.includes('oom') || m.includes('killed')) {
    return 'oom';
  }
  if (m.includes('gatherblockquantized') || m.includes('bits==4 or 8')) {
    return 'gather_quant';
  }
  if (m.includes('shader-f16') || m.includes('requires f16')) {
    return 'shader_f16';
  }
  if (m.includes('webgpu validation')) {
    return 'webgpu_validation';
  }
  if (m.includes('external data') || m.includes('mountedfiles') || m.includes('onnx_data')) {
    return 'external_data';
  }
  return 'error';
}

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

export function gemma4BrowserHtml(modelId, spec, prompt, maxNewTokens) {
  const sessionOpts = spec.sessionJson || '{}';
  const promptJson = JSON.stringify(prompt);
  return `<!DOCTYPE html><html><body><script type="module">
const prompt = ${promptJson};
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
  const inferStart = performance.now();
  const out = await generator(prompt, { max_new_tokens: ${maxNewTokens}, do_sample: false });
  window.__RESULT__ = {
    status: 'ok',
    load_ms: Math.round(loadMs),
    infer_ms: Math.round(performance.now() - inferStart),
    generated_text: out?.[0]?.generated_text ?? null,
    shader_f16: shaderF16,
    dtype_used: usedDtype,
  };
} catch (e) {
  window.__RESULT__ = { status: 'error', error: e.message };
}
</script></body></html>`;
}

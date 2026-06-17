import fs from 'node:fs/promises';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright-core';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');
const PORT = 8765;

const HTML = `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>EmbeddingGemma WebGPU probe</title></head>
<body>
<pre id="log">loading…</pre>
<script type="module">
const log = (m) => { document.getElementById('log').textContent += '\\n' + m; };
try {
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
  env.allowRemoteModels = true;
  env.useBrowserCache = true;
  log('transformers loaded');
  if (!navigator.gpu) throw new Error('navigator.gpu missing');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no WebGPU adapter');
  const info = adapter.info ?? {};
  log('adapter: ' + (info.description || info.vendor || 'ok'));
  const device = await adapter.requestDevice({
    requiredFeatures: adapter.features.has('shader-f16') ? ['shader-f16'] : [],
  });
  log('shader-f16: ' + adapter.features.has('shader-f16'));
  env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
  env.backends.onnx.webgpu.device = device;
  const t0 = performance.now();
  const extractor = await pipeline('feature-extraction', 'onnx-community/embeddinggemma-300m-ONNX', {
    dtype: 'q4f16',
    device: 'webgpu',
  });
  const loadMs = Math.round(performance.now() - t0);
  log('load ' + loadMs + 'ms');
  const t1 = performance.now();
  const out = await extractor('Stockholm är huvudstaden i Sverige.', { pooling: 'mean', normalize: true });
  const inferMs = Math.round(performance.now() - t1);
  const data = out.tolist ? out.tolist()[0] : Array.from(out.data);
  log('infer ' + inferMs + 'ms dim=' + data.length);
  log('sample=' + JSON.stringify(data.slice(0,3).map(x=>+Number(x).toFixed(6))));
  window.__RESULT__ = { status: 'ok', dim: data.length, load_ms: loadMs, infer_ms: inferMs, device: 'webgpu' };
} catch (e) {
  log('ERROR: ' + e.message);
  window.__RESULT__ = { status: 'error', error: e.message, device: 'webgpu' };
}
</script>
</body>
</html>`;

function serve() {
  return new Promise((resolve) => {
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'Content-Type': 'text/html' });
      res.end(HTML);
    });
    server.listen(PORT, '127.0.0.1', () => resolve(server));
  });
}

const server = await serve();
console.log(`Probe page: http://127.0.0.1:${PORT}/`);

const browser = await chromium.launch({
  channel: 'chrome',
  headless: true,
  args: ['--enable-unsafe-webgpu', '--use-angle=default'],
});

try {
  const page = await browser.newPage();
  page.setDefaultTimeout(600000);
  await page.goto(`http://127.0.0.1:${PORT}/`, { waitUntil: 'domcontentloaded' });
  await page.waitForFunction(() => window.__RESULT__, null, { timeout: 600000 });
  const result = await page.evaluate(() => window.__RESULT__);
  const log = await page.evaluate(() => document.getElementById('log').textContent);
  const payload = { result, log };
  const outPath = path.join(root, 'results', `probe-embeddinggemma-webgpu-browser-${Date.now()}.json`);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  await fs.writeFile(outPath, JSON.stringify(payload, null, 2));
  console.log('Wrote', outPath);
  console.log(log);
  console.log('\nResult:', JSON.stringify(result));
} finally {
  await browser.close();
  server.close();
}

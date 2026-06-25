# ONNX Runtime 1.27 — install guide

This repo pins **ONNX Runtime 1.27.0** from npm for both server (`onnxruntime-node`) and web (`onnxruntime-web`). That replaces the older ORT bundled inside `@huggingface/transformers` 4.2.0 (node **1.24.3**, web **1.26.0-dev**).

## Why 1.27?

| Benefit | Detail |
|---------|--------|
| **No source build** | `npm install` only — no `vendor/onnxruntime` clone |
| **E2B q4 CPU speed** | ~**10%** higher tok/s vs bundled 1.24 (5-run benchmark) |
| **q2f16 mobile models** | `GatherBlockQuantized` bits=2 needs ORT **1.27+** |
| **Server + web parity** | Same version for Node CPU and browser WASM/WebGPU |

See **[GEMMA4_ORT128.md](../GEMMA4_ORT128.md)** for benchmark numbers.

## Quick start

From the repo root:

```bash
npm install
node -e "import('onnxruntime-node').then(m => console.log('ORT', m.env.versions?.common))"
# → ORT 1.27.0
```

`package.json` already lists the pins and overrides — a normal `npm install` is enough.

## What `package.json` does

Direct dependencies (so your app resolves 1.27):

```json
{
  "dependencies": {
    "@huggingface/transformers": "^4.2.0",
    "onnxruntime-common": "1.27.0",
    "onnxruntime-node": "1.27.0",
    "onnxruntime-web": "1.27.0"
  }
}
```

**Overrides** force Transformers.js to use the same ORT instead of its nested 1.24/1.26 copies:

```json
{
  "overrides": {
    "@huggingface/transformers": {
      "onnxruntime-common": "1.27.0",
      "onnxruntime-node": "1.27.0",
      "onnxruntime-web": "1.27.0"
    },
    "onnxruntime-common": "1.27.0",
    "onnxruntime-node": "1.27.0",
    "onnxruntime-web": "1.27.0"
  }
}
```

Copy this block into any project that uses `@huggingface/transformers` with Gemma 4.

## Verify install

```bash
# Node ORT version
node -e "import('onnxruntime-node').then(m => console.log(m.env.versions?.common))"

# Web ORT version
node -e "import('node:fs').then(fs=>console.log(JSON.parse(fs.readFileSync('node_modules/onnxruntime-web/package.json','utf8')).version))"

# E2B q4 CPU smoke (downloads model on first run)
node --expose-gc scripts/benchmark-gemma4-variant.mjs \
  --model-id onnx-community/gemma-4-E2B-it-ONNX \
  --model-slug E2B-it --model-name "Gemma 4 E2B IT" \
  --dtype q4 --backend cpu --max-prompts 1

# q2f16 gather smoke (needs cached mobile ONNX)
npm run verify:ort:q2f16
npm run verify:ort:web:q2f16
```

## Using ORT 1.27 in code

### Server — CPU (recommended for E2B q4)

```javascript
import { createTextGenerator } from './lib/transformers-runtime.mjs';

const gen = await createTextGenerator(
  'onnx-community/gemma-4-E2B-it-ONNX',
  { dtype: 'q4' },
  'cpu',
);
```

### Server — WASM-JSEP

```javascript
const gen = await createTextGenerator(
  'onnx-community/gemma-4-E2B-it-ONNX',
  { dtype: 'q4' },
  'wasm-jsep',  // not 'wasm' — asyncify lacks GatherBlockQuantized for q4
);
```

### Browser — WebGPU

```javascript
import { pipeline, env } from '@huggingface/transformers';

const adapter = await navigator.gpu.requestAdapter();
const device = await adapter.requestDevice();
env.backends.onnx.webgpu = { device };

const gen = await pipeline('text-generation', 'onnx-community/gemma-4-E2B-it-ONNX', {
  dtype: 'q4',
  device: 'webgpu',
});
```

For WASM in the browser, copy `ort-wasm-simd-threaded.jsep.*` from `node_modules/onnxruntime-web/dist/` to your static assets and set `env.backends.onnx.wasm.wasmPaths` if needed.

## Fresh project (minimal `package.json`)

```bash
npm init -y
npm install @huggingface/transformers@^4.2.0 \
  onnxruntime-common@1.27.0 \
  onnxruntime-node@1.27.0 \
  onnxruntime-web@1.27.0
```

Then add the `overrides` block above before running inference.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Still on ORT 1.24 | Add `overrides` for `@huggingface/transformers`; delete `node_modules` and `package-lock.json`, then `npm install` |
| `Cannot find module 'onnxruntime-node'` | Add `onnxruntime-node` as a direct dependency (not only nested under transformers) |
| `gather_quant` on WASM | Use backend `wasm-jsep`, not `wasm` |
| q2f16 fails | Confirm `node -e "…"` prints **1.27.0** |
| Need ORT newer than npm | Optional local build: `npm run build:ort:all` — see [gemma4-q2f16.md](./gemma4-q2f16.md) |

## Optional: build from source

Only if you need unreleased ORT `main` (e.g. 1.28 pre-release). Most users should stay on **npm 1.27.0**.

```bash
npm run build:ort:all
# then switch package.json deps to file:vendor/onnxruntime/js/…
ONNXRUNTIME_NODE_INSTALL=skip npm install
```

Build scripts remain in `scripts/build-ort.sh` and `scripts/build-ort-web.sh`.
